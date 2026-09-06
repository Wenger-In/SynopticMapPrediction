from keras.layers import Dense,Input
import tensorflow
from keras import Model
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from smp.config import load_config
from smp.utils import min_max_normalize, set_random_seed
 
# Encoder
def get_encoder(in_shape,out_units):
    model = Sequential()
    model.add(Dense(500,input_shape=(in_shape,), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(out_units, activation='relu'))
    model.summary()
    return model
    
# Decoder
def get_decoder(in_shape,out_units):
    model = Sequential()
    model.add(Dense(100,input_shape=(in_shape,), activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(out_units, activation='sigmoid'))
    return model
def get_autoencoder():
    # Autoencoder
    # data_num=784
    data_num=73*30
    # feature_num=2
    feature_num=10
    #定义编码器的输入
    ae_input = Input(shape=(data_num,), name="AE_input")
    
    #获取编码器
    encoder=get_encoder(data_num,feature_num)
    #获取解码器
    decoder=get_decoder(feature_num,data_num)
    
    #使用编码器编码输入得到特征
    ae_encoder_output = encoder(ae_input)
    #使用解码器解码特征得到输出
    ae_decoder_output = decoder(ae_encoder_output)
    #构造自动解码器模型
    ae = Model(ae_input, ae_decoder_output, name="AE")
    ae.summary()
    #编译自动解码器
    ae.compile(loss="binary_crossentropy", optimizer='adam')
    return ae,encoder,decoder
def get_mnist_data():
    # Preparing MNIST Dataset
    (x_train_orig, y_train), (x_test_orig, y_test) = tensorflow.keras.datasets.mnist.load_data()
    x_train_orig = x_train_orig.astype("float32") / 255.0
    x_test_orig = x_test_orig.astype("float32") / 255.0
    
    x_train = x_train_orig.reshape((-1, np.prod(x_train_orig.shape[1:])))
    x_test = x_test_orig.reshape((-1, np.prod(x_test_orig.shape[1:])))
    return x_train,y_train,x_test,y_test
def get_map_data(folder_path):
    # Import WSO synoptic maps
    file_pattern = re.compile(r'^cr(\d{4})\.dat$')
    data_lst = []
    cr_lst = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            match = file_pattern.match(file)
            if match:
                file_path = os.path.join(root, file)
                try:
                    data = np.loadtxt(file_path)
                    data = data[np.newaxis, :, :]
                    data_lst.append(data)
                    cr_lst.append(int(match.group(1)))
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    data_con_orig = np.stack(data_lst, axis=0)
    data_con = min_max_normalize(data_con_orig)
    numbers = np.array(cr_lst)
    
    split_index = int(0.8 * len(data_lst))
    data_train = data_con[:split_index, :, :]
    data_test = data_con[split_index:, :, :]
    cr_train = numbers[:split_index]
    cr_test = numbers[split_index:]
    
    data_train = data_train.reshape((-1, np.prod(data_train.shape[1:])))
    data_test = data_test.reshape((-1, np.prod(data_test.shape[1:])))
    return data_train, cr_train, data_test, cr_test
def main():
    """Train and inspect an autoencoder for WSO synoptic maps."""
    config = load_config()
    set_random_seed(config.random_seed)
    ae, encoder, decoder = get_autoencoder()
    x_train, y_train, x_test, _ = get_map_data(config.path("wso_field"))
    ae.fit(
        x_train,
        x_train,
        epochs=500,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, x_test),
    )
    encoded_images = encoder.predict(x_train)
    decoded_images = decoder.predict(encoded_images)
    decoded_images_orig = np.reshape(decoded_images, (-1, 30, 73))

    num_images_to_show = min(5, x_train.shape[0])
    for im_ind in range(num_images_to_show):
        plot_ind = im_ind * 2 + 1
        rand_ind = np.random.randint(low=0, high=x_train.shape[0])
        plt.subplot(num_images_to_show, 2, plot_ind)
        plt.imshow(x_train[rand_ind].reshape((30, 73)), cmap="RdBu")
        plt.colorbar()
        plt.subplot(num_images_to_show, 2, plot_ind + 1)
        plt.imshow(decoded_images_orig[rand_ind], cmap="RdBu")
        plt.colorbar()

    plt.figure()
    plt.scatter(encoded_images[:, 0], encoded_images[:, 1], c=y_train)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
