clear; close all;
cfg = project_config();
path = [fullfile(cfg.output_root, 'predict_SC25', 'comparison'), filesep];
cr = 2259;
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
obs_name = ['cr',num2str(cr),'_obs.mat'];
pred_name = ['cr',num2str(cr),'_pred.mat'];
obs_map = load([path, obs_name]);
obs_map = obs_map.magneto;
pred_map = load([path, pred_name]);
pred_map = pred_map.magneto;
%% calculate residual
res_map = pred_map - obs_map;
res_map = flip(res_map,1);
%% figure
figure();
LineWidth = 2;
FontSize = 15;

% p = pcolor(log10(abs(res_map./obs_map)));
% clim = max(max(abs(log10(abs(res_map./obs_map)))));
p = pcolor(res_map);
% p = pcolor(abs(res_map));
set(p, 'LineStyle', 'none')
% contour(abs(res_map),[1,1],'ShowText','on');
hold on
contour(res_map,[1.5,-1.5],'k','ShowText','on','LineWidth',1);
hold on
contour(res_map,[0.5,-0.5],'--k','ShowText','on','LineWidth',1);
clim = max(max(obs_map));
cb = colorbar;
axis equal
cb.Title.String = '[G]';
colormap(red_white_blue);
% hold on
% yline(90,'--k','LineWidth',LineWidth);
axis equal
xlim([0 360]);
ylim([0 180]);
xticks([0 90 180 270 360]);
xticklabels({'0^\circ','90^\circ','180^\circ','270^\circ','360^\circ'})
yticks([0 45 90 135 180]);
yticklabels({'-90^\circ','-45^\circ','0^\circ','45^\circ','90^\circ'})
set(gca,'Clim',[-clim,clim],'TickDir','out','XminorTick','on','YminorTick','on','LineWidth',LineWidth,'FontSize',FontSize);
