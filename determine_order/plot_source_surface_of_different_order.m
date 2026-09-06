clear; close all;
cfg = project_config();
path = [fullfile(cfg.repo_root, 'determine_order'), filesep];
gong_file = '2239_gong_pfss.mat';
WSO_5_file = '2239_WSO_5_pfss.mat';
WSO_9_file = '2239_WSO_9_pfss.mat';
%% colorbar red-white-blue
color_red   = [1,0,0];
color_white = [1,1,1];
color_blue  = [0,0,1];
n1 = 100;
n2 = 100;
R_comp = [linspace(color_red(1),color_white(1),n1),linspace(color_white(1),color_blue(1),n2)];
G_comp = [linspace(color_red(2),color_white(2),n1),linspace(color_white(2),color_blue(2),n2)];
B_comp = [linspace(color_red(3),color_white(3),n1),linspace(color_white(3),color_blue(3),n2)];
red_white_blue = [R_comp',G_comp',B_comp'];
%% import data
Br_gong = load([path, gong_file]);
Br_WSO_5 = load([path, WSO_5_file]);
Br_WSO_9 = load([path, WSO_9_file]);
%% extract data
Br_gong = Br_gong.data;
Br_WSO_5 = Br_WSO_5.data;
Br_WSO_9 = Br_WSO_9.data;
%% figures
figure(1);
p1 = pcolor(Br_WSO_5);
set(p1, 'LineStyle', 'none')
plot_porperties(Br_WSO_5, red_white_blue)

figure(2);
p2 = pcolor(Br_WSO_9);
set(p2, 'LineStyle', 'none')
plot_porperties(Br_WSO_9, red_white_blue)

figure(3);
p3 = pcolor(Br_gong);
set(p3, 'LineStyle', 'none')
plot_porperties(Br_gong, red_white_blue)
%% calculate correlation coefficient
cc_5 = corrcoef(Br_WSO_5, Br_gong);
cc_5 = cc_5(1,2);
cc_9 = corrcoef(Br_WSO_9, Br_gong);
cc_9 = cc_9(1,2);
cc_59 = corrcoef(Br_WSO_5, Br_WSO_9);
cc_59 = cc_59(1,2);

function plot_porperties(Br, red_white_blue)
    LineWidth = 2;
    FontSize = 15;
    cb = colorbar;
    cb.Title.String = '[G]';
    colormap(red_white_blue);
    hold on
    yline(90,'--k','LineWidth',LineWidth);
    axis equal
    xlim([0 360]);
    ylim([0 180]);
    xticks([0 90 180 270 360]);
    xticklabels({'0^\circ','90^\circ','180^\circ','270^\circ','360^\circ'})
    yticks([0 45 90 135 180]);
    yticklabels({'-90^\circ','-45^\circ','0^\circ','45^\circ','90^\circ'})
    clim = max(max(abs(Br)));
    set(gca,'CLim',[-clim clim],'TickDir','out','XminorTick','on','YminorTick','on','LineWidth',LineWidth,'FontSize',FontSize);
end

