from typing import List, Tuple, Dict, Mapping, Any
import torch
import numpy as np
import numpy.typing as npt
import sys
import math
import svgwrite
import webbrowser
import os

from smp.config import load_config
from smp.utils import set_random_seed

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        inputs = 1 # 输入数组大小（时间）
        hidden = 64 # 隐藏层数组大小
        outputs = 1 # 输出数组大小（物理量）
        self.affine1 = torch.nn.Linear(inputs, hidden) # 输入层到隐藏层的（正余弦单元）连接
        self.affine2 = torch.nn.Linear(hidden, outputs) # 隐藏层到输出层的（正余弦单元）连接
        self.augment = torch.nn.Linear(inputs, outputs) # 输入层到输出层的（增强单元）连接

    def init_weights(self) -> None: # 网络初始化
        m = self.affine1.bias.shape[0] // 2 # 前一半是sin单元，后一半是cos单元
        if m * 2 != self.affine1.bias.shape[0]:
            raise ValueError('Expected an even number of units') # 需要偶数个正余弦单元
        for i in range(m):
            # sine unit
            self.affine1.weight.data[i] = 2 * math.pi * i # sin单元权重初始化
            self.affine1.bias.data[i] = 0. # 前半部分是sin单元

            # cosine unit
            self.affine1.weight.data[m + i] = 2 * math.pi * i # cos单元权重初始化
            self.affine1.bias.data[m + i] = math.pi / 2 # 后半部分是cos单元

    def forward(self, t:torch.Tensor) -> torch.Tensor:
        return self.affine2(torch.sin(self.affine1(t))) + self.augment(t) # type: ignore # 正余弦单元和增强单元相加，作为输出

class CustomLoss(torch.nn.Module): # 计算预测值与实际值之间的均方误差
    def __init__(self, model:Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, preds:torch.Tensor, targs:torch.Tensor) -> torch.Tensor:
        regularize = 1e-2 # 正则项
        return torch.mean((targs - preds) ** 2) + regularize * torch.sum(torch.abs(self.model.affine2.weight)) # 计入正则化的影响

def train(x:torch.Tensor, y:torch.Tensor) -> Model:
    if x.shape[0] != y.shape[0]:
        raise ValueError('Expected the same number of feature-vectors and label-vectors') # 保证时间与物理量的序列长度一致

    # Determine what device to train on
    if torch.cuda.is_available():
        device_name = 'cuda' # Compute Unified Device Architecture
    elif torch.backends.mps.is_available():
        device_name = 'mps' # Metal Performance Shaders
    else:
        device_name = 'cpu' # Central Processing Unit
    print(f'Training on {device_name}')
    device = torch.device(device_name)

    # Make a model
    x = x.to(device)
    y = y.to(device)
    model = Model()
    model.to(device)
    loss_fn = CustomLoss(model) # 用于计算模型预测值与真实值的误差
    loss_fn.to(device)
    zero_loss = torch.zeros((1,)).to(device)

    # Make an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Adam优化器，学习率为1e-3

    # Train
    batch_size = min(x.shape[0], 32) # 批量放入训练数据
    for epoch in range(10000): # 训练10000步，没有设置早停
        model.train() # switch the model to training mode
        all_indexes = torch.randperm(x.shape[0]) # 打散数据顺序
        batch_start = 0
        sum_loss = zero_loss
        while batch_start + batch_size <= x.shape[0]:

            # Make a batch
            batch_indexes = all_indexes[batch_start:batch_start + batch_size] # 随机分批，用于批量输入
            batch_x = x[batch_indexes]
            batch_y = y[batch_indexes]

            # Refine the model
            predictions = model(batch_x) # 输入该批次，进行模型预测
            loss = loss_fn(predictions, batch_y) # 计算模型预测值与真实值的误差
            sum_loss = sum_loss + loss # 累积误差
            optimizer.zero_grad() # 梯度置零
            loss.backward() # 误差反向传播
            optimizer.step()

            # Advance to the next batch
            batch_start += batch_size # 输入下一批
        if epoch % 100 == 0:
            print(f'epoch={epoch}, loss={sum_loss.cpu().detach().item()}') # 每100轮训练输出一次误差，用于监测

    model.to('cpu')
    return model


class Plotter(): # 画图部分，不赘述
    def __init__(self, size:Tuple[int, int], bottom_left:Tuple[float, float], top_right:Tuple[float, float]) -> None:
        self.d = svgwrite.Drawing('untitled.svg', size=size, profile='full')
        self.size = size
        self.mins = bottom_left
        self.maxs = top_right

    def _proj(self, x: Tuple[float,float]) -> Tuple[float,float]:
        return (
            (x[0] - self.mins[0]) / (self.maxs[0] - self.mins[0]) * self.size[0],
            self.size[1] - (x[1] - self.mins[1]) / (self.maxs[1] - self.mins[1]) * self.size[1],
        )

    def line(self, a:Tuple[float,float], b:Tuple[float,float], thickness:float=1., color:Tuple[int,int,int]=(0,0,0), absolute:bool=False) -> None:
        if not absolute:
            a = self._proj(a)
            b = self._proj(b)
        self.d.add(self.d.line(a, b, stroke_width=thickness, stroke=svgwrite.rgb(*color)))

    def arrow(self, a:Tuple[float,float], b:Tuple[float,float], thickness:float=1., color:Tuple[int,int,int]=(0,0,0), head_size:float=10., head_angle:float=0.785398163, absolute:bool=False) -> None:
        angle = math.atan2(a[1] - b[1], a[0] - b[0])
        ang1 = angle + head_angle / 2.
        ang2 = angle - head_angle / 2.
        self.line(a, b, thickness, color, absolute)
        self.line(b, (b[0] + head_size * math.cos(ang1), b[1] + head_size * math.sin(ang1)), thickness, color, absolute)
        self.line(b, (b[0] + head_size * math.cos(ang2), b[1] + head_size * math.sin(ang2)), thickness, color, absolute)

    def rect(self, a:Tuple[float,float], b:Tuple[float,float], color:Tuple[int,int,int]=(0,0,0), absolute:bool=False) -> None:
        if not absolute:
            a = self._proj(a)
            b = self._proj(b)
        tl = (min(a[0], b[0]), min(a[1], b[1]))
        br = (max(a[0], b[0]), max(a[1], b[1]))
        wh = (br[0]-tl[0],br[1]-tl[1])
        self.d.add(self.d.rect(tl, wh, stroke='none', fill=svgwrite.rgb(*color)))

    def circle(self, pos:Tuple[float,float], radius:float=1., color:Tuple[int,int,int]=(0,0,0), absolute:bool=False) -> None:
        if not absolute:
            pos = self._proj(pos)
        self.d.add(self.d.circle(pos, r=radius, fill=svgwrite.rgb(*color)))

    def text(self, s:str, pos:Tuple[float,float], size:float=16, color:Tuple[int,int,int]=(0,0,0), absolute:bool=False) -> None:
        if not absolute:
            pos = self._proj(pos)
        if not np.isnan(pos[0]) and not np.isnan(pos[1]):
            self.d.add(self.d.text(s, insert=pos, fill=svgwrite.rgb(*color), style=f'font-size:{size}px; font-family:Arial'))

    def tostr(self) -> str:
        return '<?xml version="1.0" encoding="utf-8" ?>\n' + str(self.d.tostring())

    def tosvg(self, filename:str) -> None:
        with open(filename, 'w') as f:
            f.write(self.tostr())


if __name__ == '__main__':
    set_random_seed(load_config().random_seed)
    print('Training...')
    train_x = torch.arange(100, dtype=torch.float).reshape((-1, 1)) # 真实时间序列
    train_y = (5 * torch.sin(train_x / 5) + train_x / 5).reshape((-1, 1)) # 真实物理量序列
    model = train(train_x, train_y) # 训练过程

    print('Plotting...')
    test_x = torch.arange(200) # 需要预测的时间序列
    p = Plotter(size=(800, 600), bottom_left=(-1., -1.), top_right=(201., 50.))
    for i in range(1, test_x.shape[0]):
        a = (test_x[i - 1].item(), model(torch.Tensor([[test_x[i - 1]]]).to(torch.float)).cpu().detach().item()) # 从torch.tensor中提取出来
        b = (test_x[i].item(), model(torch.Tensor([[test_x[i]]]).to(torch.float)).cpu().detach().item()) # 从torch.tensor中提取出来
        p.line(a, b, thickness=1., color=(0, 0, 128)) # 蓝色，预测数据
    for i in range(train_x.shape[0]):
        p.circle((train_x[i].item(), train_y[i].item()), radius = 2., color=(128, 0, 0)) # 红色，训练数据
    with open('plot.svg', 'w') as f:
        f.write(p.tostr())
    filename = f'file:///{os.getcwd()}/plot.svg/'
    webbrowser.open(filename, new=2)
