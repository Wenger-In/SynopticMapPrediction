clear; close all;
cfg = project_config();
%% import data
pred_dir = [fullfile(cfg.paths.sunspot_output, 'smooth_prediction', '0'), filesep];
future = importdata([pred_dir,'future_predict.csv']);

hist_dir = [cfg.paths.sunspot_root, filesep];
ind_ms_end = 3303;
ind_m_end = 3309;
sn_ms_info = importdata([hist_dir,'SN_ms_tot_V2.0_202410.csv']);
sn_ms = sn_ms_info(2727:ind_ms_end,4); % Line 2727-3287
sn_m_info = importdata([hist_dir,'SN_m_tot_V2.0_202410.csv']);
sn_m = sn_m_info(2727:ind_m_end,4); % Line 2727-3304

future_step = 144;
index = 1:length(sn_ms) + future_step;
%% Component: colormap stand for Solar Cycle
year_lst_m = sn_m_info(2727:ind_m_end,3);
year_lst_ms = sn_m_info(2727:ind_ms_end,3);
year_beg = year_lst_m(1);
year_end_m = year_lst_m(end);
year_end_ms = year_lst_ms(3287-2727+1);

year_gap_1 = sn_m_info(2853,3);
year_gap_2 = sn_m_info(2971,3);
year_gap_3 = sn_m_info(3120,3);
year_gap_4 = sn_m_info(3252,3);
n_1 = 2853-2727;
n_2 = 2971-2853;
n_3 = 3120-2971;
n_4 = 3252-3120;                 
n_5 = ind_m_end-3252;
full_num = max([n_1,n_2,n_3,n_4,n_5]);
color_sc_1 = [1,0,0];
color_sc_2 = [0,1,0];
color_sc_3 = [0,0,1];
color_sc_4 = [1,0,1];
color_sc_5 = [0,0,0];
color_white = [1,1,1];
sc_full_1 = [linspace(color_sc_1(1),color_white(1),full_num); linspace(color_sc_1(2),color_white(2),full_num); linspace(color_sc_1(3),color_white(3),full_num)];
sc_full_2 = [linspace(color_sc_2(1),color_white(1),full_num); linspace(color_sc_2(2),color_white(2),full_num); linspace(color_sc_2(3),color_white(3),full_num)];
sc_full_3 = [linspace(color_sc_3(1),color_white(1),full_num); linspace(color_sc_3(2),color_white(2),full_num); linspace(color_sc_3(3),color_white(3),full_num)];
sc_full_4 = [linspace(color_sc_4(1),color_white(1),full_num); linspace(color_sc_4(2),color_white(2),full_num); linspace(color_sc_4(3),color_white(3),full_num)];
sc_full_5 = zeros(3,n_5) + color_sc_5.';
colormap_sc = [sc_full_1(:,1:n_1).'; sc_full_2(:,1:n_2).'; sc_full_3(:,1:n_3).'; sc_full_4(:,1:n_4).'; sc_full_5.'];
%% plot figure
figure();
LineWidth = 2;
FontSize = 20;
sz = 100;

% plot smoothed SSN
plot(year_lst_m(1:2853-2726), sn_ms(1:2853-2726),'r','LineWidth',LineWidth*2)
hold on
plot(year_lst_m(2853-2726:2971-2726), sn_ms(2853-2726:2971-2726),'g','LineWidth',LineWidth*1.5)
hold on
plot(year_lst_m(2971-2726:3120-2726), sn_ms(2971-2726:3120-2726),'b','LineWidth',LineWidth*1.5)
hold on
plot(year_lst_m(3120-2726:3252-2726), sn_ms(3120-2726:3252-2726),'m','LineWidth',LineWidth*1.5)
hold on
plot(year_lst_m(3252-2726:ind_ms_end-2726), sn_ms(3252-2726:ind_ms_end-2726),'k','LineWidth',LineWidth*1.5)
hold on

% plot predicted SSN
plot(year_end_ms+(1:future_step)/12,future,'c','LineWidth',LineWidth*1.5);
hold on

% plot unsmoothed SSN
% plot(year_lst,sn_m,'k','LineWidth',LineWidth/2);
hold on
scatter(year_lst_m,sn_m,sz,year_lst_m,'filled');
hold on
colormap(colormap_sc);

box off
xlabel('YEAR')
ylabel('Sunspot Number')
xlim([year_beg, 2035])
set(gca,'LineWidth',LineWidth, 'FontSize', FontSize, ...
    'XMinorTick','on','YMinorTick','on','XColor','k','Ycolor','k')





