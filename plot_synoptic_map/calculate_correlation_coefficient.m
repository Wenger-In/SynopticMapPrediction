clear; close all;
cfg = project_config();
cr = 2261;
pred_or_pers = 0; % 0-predict model; 1-persistent model
%% import data
obs_dir = [fullfile(cfg.output_root, 'harmonics_map', 'WSO'), filesep];
obs_file = [obs_dir,'cr',num2str(cr),'_obs.mat'];
obs_data = load(obs_file);
magneto_obs = obs_data.magneto;
%% comparison
if pred_or_pers == 0
    pred_dir = [fullfile(cfg.output_root, 'predict_SC25'), filesep];
    pred_file = [pred_dir,'cr',num2str(cr),'_pred.mat'];
    pred_data = load(pred_file);
    magneto_pred = pred_data.magneto;
elseif pred_or_pers == 1
    pers_file = [obs_dir,'cr2258_obs.mat'];
    pers_data = load(pers_file);
    magneto_pers = pers_data.magneto;
    magneto_pred = magneto_pers;
end
%% interpolate into rougher grid
raw_lon = linspace(0,360,360);
raw_lat = linspace(-90,90,180);
[raw_llon,raw_llat] = meshgrid(raw_lon,raw_lat);
std_lon = linspace(0,360,73);
std_lat = linspace(-90,90,37);
% std_lat = linspace(-75,75,37);
[std_llon,std_llat] = meshgrid(std_lon,std_lat);

Br_obs = interp2(raw_llon,raw_llat,magneto_obs,std_llon,std_llat);
Br_pred = interp2(raw_llon,raw_llat,magneto_pred,std_llon,std_llat);
%% calculate correlation coefficient
Br_obs = reshape(Br_obs,1,[]);
Br_pred = reshape(Br_pred,1,[]);
cc = corrcoef(Br_obs,Br_pred);
p = polyfit(Br_obs,Br_pred,1);
xFit = linspace(min(Br_obs), max(Br_obs), 100);
yFit = polyval(p, xFit);
%% plot scatters and counts
binwidth = 0.2;
binx = -4+binwidth/2:binwidth:4-binwidth/2;
biny = -4+binwidth/2:binwidth:4-binwidth/2;
[binxx,binyy] = meshgrid(binx,biny);
[gridcounts,~,~,~,~] = histcounts2(Br_pred,Br_obs,'XBinLimits',[-4,4],'YBinLimits',[-4,4],'BinWidth',binwidth);

figure();

subplot(1,2,1)
scatter(Br_obs,Br_pred,3);
hold on;
plot(xFit,yFit,'r','LineWidth',2);
grid on
axis equal
xlim([-4,4])
ylim([-4,4])
xlabel('Observation [G]')
ylabel('Prediction [G]')
set(gca,'LineWidth',2,'FontSize',15)

subplot(1,2,2)
p0 = pcolor(binxx,binyy,gridcounts);
set(p0,'LineStyle','none')
hold on;
plot(binx,biny,'w','LineWidth',2);
colormap('jet');
cb = colorbar;
cb.Label.String = 'count';
axis equal
xlim([-4,4])
ylim([-4,4])
xlabel('Observation (G)')
ylabel('Prediction (G)')
set(gca,'LineWidth',2,'FontSize',15,'CLim',[0,50],'TickDir','out')

sgtitle(['CR',num2str(cr),',k=',num2str(p(1)),',b=',num2str(p(2)),',CC=',num2str(cc(1,2))],'FontSize',30);

annotation('textbox',[0.6,0,0.1,0.05],'String','plotted by calculate\_correlation\_coefficient.m','FitBoxToText','on');
%% plot histcounts
% edges = -4:0.5:4;
% [counts_obs, ~] = histcounts(Br_obs,edges);
% [counts_pred,~] = histcounts(Br_pred,edges);
% 
% figure()
% histogram(Br_obs,edges,'EdgeColor','r','FaceColor','none'); hold on
% histogram(Br_pred,edges,'EdgeColor','b','FaceColor','none'); hold on
%% illustrate cycle-like scatters
% test1 = cos(std_llon/60).*cos(std_llat/60);
% test2 = 0.6*cos(std_llon/60-0.5).*cos(std_llat/60);
% 
% figure()
% subplot(2,2,1)
% p1 = pcolor(test1);
% colorbar
% set(p1,'LineStyle','none')
% subplot(2,2,2)
% p2 = pcolor(test2);
% colorbar
% set(p2,'LineStyle','none')
% subplot(2,2,3)
% test1 = reshape(test1(1,:),1,[]);
% test2 = reshape(test2(1,:),1,[]);
% scatter(test1,test2,3)
% annotation('textbox',[0.6,0,0.1,0.05],'String','plotted by calculate\_correlation\_coefficient.m','FitBoxToText','on');
