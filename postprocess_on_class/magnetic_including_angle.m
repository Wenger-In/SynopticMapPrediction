clear; close all;
cfg = project_config();
data_dir = [fullfile(cfg.repo_root, 'postprocess_on_class', 'earth_location'), filesep];
swmf_path = [data_dir, 'SWMF_1AU_Bn_Br_Bt_2538.xlsx'];
omni_2022_path = [data_dir, 'OMNI_data_2022_to_ss.xlsx'];
omni_2023_path = [data_dir, 'OMNI_data_2023_to_ss.xlsx'];
%% import data
swmf_data = readmatrix(swmf_path);
omni_2022_data = readmatrix(omni_2022_path);
omni_2023_data = readmatrix(omni_2023_path);
omni_data = [omni_2022_data;omni_2023_data];
%% convert datetime to epoch
num_swmf = length(swmf_data);
num_omni = length(omni_data);
swmf_epoch = zeros(num_swmf,1);
omni_epoch = zeros(num_omni,1);
for i_swmf = 1 : num_swmf
    swmf_datetime = datetime([num2str(swmf_data(i_swmf,1)),'-',num2str(swmf_data(i_swmf,2)),'-', ...
        num2str(swmf_data(i_swmf,3)),' ',num2str(swmf_data(i_swmf,4))], ...
        'InputFormat', 'yyyy-MM-dd HH');
    swmf_epoch(i_swmf) = datenum(swmf_datetime);
end
for i_omni = 1 : num_omni
    omni_datetime = datetime([num2str(omni_data(i_omni,1)),'-',num2str(omni_data(i_omni,15)),'-', ...
        num2str(omni_data(i_omni,16)),' ',num2str(omni_data(i_omni,3))], ...
        'InputFormat', 'yyyy-MM-dd HH');
    omni_epoch(i_omni) = datenum(omni_datetime);
end
%% eliminate bad pointst
bad_beg = datenum('2022-10-11');
bad_end = datenum('2022-12-04');
swmf_bad_indices = find(swmf_epoch>bad_beg & swmf_epoch<bad_end);
omni_bad_indices = find(omni_epoch>bad_beg & omni_epoch<bad_end);
swmf_epoch(swmf_bad_indices) = [];
omni_epoch(omni_bad_indices) = [];
swmf_data(swmf_bad_indices,:) = [];
omni_data(omni_bad_indices,:) = [];
%% select common epoch
shared_epoch = intersect(swmf_epoch, omni_epoch);
swmf_indices = find(ismember(swmf_epoch, shared_epoch));
omni_indices = find(ismember(omni_epoch, shared_epoch));
swmf_epoch = swmf_epoch(swmf_indices);
swmf_shared = swmf_data(swmf_indices,:);
omni_epoch = omni_epoch(omni_indices);
omni_shared = omni_data(omni_indices,:);
%% extract Brtn in common epoch
swmf_Brtn = swmf_shared(:,[10,11,9]);
omni_Brtn = omni_shared(:,[6,7,8]);
%% calculate including angle
dot_Brtn = dot(swmf_Brtn,omni_Brtn,2);
swmf_norm = sqrt(sum(swmf_Brtn.^2, 2));
omni_norm = sqrt(sum(omni_Brtn.^2, 2));
inc_angle = acos(dot_Brtn./ (swmf_norm.*omni_norm));
inc_angle = rad2deg(inc_angle);
%% cheery picking
% slt_frac = round(6/24,3);
% mod_epoch = mod(shared_epoch,1);
% mod_epoch = round(mod_epoch,3);
% slt_indices = find(mod_epoch == slt_frac);
% slt_epoch = shared_epoch(slt_indices);
% slt_inc_angle = inc_angle(slt_indices);
% inc_angle = slt_inc_angle;
slt_epoch = shared_epoch;
slt_inc_angle = inc_angle;
%% time node
epoch_2259 = datenum('2022-06-24'); % CR 2259 begins, prediction begins
epoch_2260 = datenum('2022-07-21'); % CR 2260 begins, 1st month prediction ends
epoch_2261 = datenum('2022-08-18'); % CR 2261 begins
epoch_2262 = datenum('2022-09-14'); % CR 2262 begins, 1st 3-month prediction ends
obs_indices = find(slt_epoch<epoch_2259);
pred_indices = find(slt_epoch>=epoch_2259);
pred_1mon_indices = find(slt_epoch>=epoch_2259 & slt_epoch<epoch_2260);
pred_3mon_indices = find(slt_epoch>=epoch_2259 & slt_epoch<epoch_2262);
%% dividing into observation and prediction
obs_inc_angle = inc_angle(obs_indices);
pred_inc_angle = inc_angle(pred_indices);
pred_1mon_inc_angle = inc_angle(pred_1mon_indices);
pred_3mon_inc_angle = inc_angle(pred_3mon_indices);
%% count including angle < 90
acc_obs = round(sum(obs_inc_angle<=90)/length(obs_inc_angle),3);
acc_pred = round(sum(pred_inc_angle<=90)/length(pred_inc_angle),3);
acc_pred_1mon = round(sum(pred_1mon_inc_angle<=90)/length(pred_1mon_inc_angle),3);
acc_pred_3mon = round(sum(pred_3mon_inc_angle<=90)/length(pred_3mon_inc_angle),3);
%% figure 1: including angle distribution
figure()
LineWidth = 2;
FontSize = 15;

subplot(1,4,1)
histogram(obs_inc_angle,'Binedges',linspace(0,180,31),'LineWidth',LineWidth)
subplot_property()
title('Observation interval','FontSize',FontSize*1.5)
subplot(1,4,2)
histogram(pred_inc_angle,'Binedges',linspace(0,180,31),'LineWidth',LineWidth)
subplot_property()
title('Prediction interval','FontSize',FontSize*1.5)
subplot(1,4,3)
histogram(pred_1mon_inc_angle,'Binedges',linspace(0,180,31),'LineWidth',LineWidth)
subplot_property()
title('Fisrt CR prediction interval','FontSize',FontSize*1.5)
subplot(1,4,4)
histogram(pred_3mon_inc_angle,'Binedges',linspace(0,180,31),'LineWidth',LineWidth)
subplot_property()
title('First 3-CR prediction interval','FontSize',FontSize*1.5)
%% figure 2: time series
figure()
plot(shared_epoch,omni_Brtn(:,1))
datetick('x','YYYY/mm/DD');
%% function
function subplot_property()
    LineWidth = 2;
    FontSize = 15;
    xticks(linspace(0,180,7))
    xlabel('Including Angle [deg.]')
    ylabel('Counts')
    grid on
    set(gca,'LineWidth',LineWidth,'FontSize',FontSize)
end
