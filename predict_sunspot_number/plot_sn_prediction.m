clear; close all;
cfg = project_config();

dev = -132;
data_dir = [fullfile(cfg.paths.sunspot_output, 'smooth_prediction', num2str(dev)), filesep];
train = importdata([data_dir,'train_predict.csv']);
val = importdata([data_dir,'val_predict.csv']);
test = importdata([data_dir,'test_predict.csv']);
future = importdata([data_dir,'future_predict.csv']);
future_llim = importdata([data_dir,'future_predict_llim.csv']);
future_ulim = importdata([data_dir,'future_predict_ulim.csv']);

sn_dir = [cfg.paths.sunspot_root, filesep];
sn_info = importdata([sn_dir,'SN_ms_tot_V2.0.csv']);
sn = sn_info(:,4);
sn(sn==-1) = nan;
test(1) = val(end);
test(2) = val(end);

look_back = 356;
future_step = 144;
index = 1:length(sn) + future_step;
epoch = 1749.042 + (index - 1)/12;
%% devide into three sets
ind_train = look_back+1:length(train)+look_back;
ind_val = look_back+length(train)+1:length(val)+length(train)+look_back;
ind_test = look_back+length(train)+length(val)+1:length(test)+length(train)+length(val)+look_back;
ind_train = ind_train + 6;
ind_val = ind_val + 6;
ind_test = ind_test + 6;
%% calculate RMSE
rmse_train = rmse(sn(ind_train), train);
rmse_val = rmse(sn(ind_val), val);
rmse_test = rmse(sn(ind_test), test);
rmse_pers_train = rmse(sn(ind_train), sn(ind_train-1));
rmse_pers_val = rmse(sn(ind_val), sn(ind_val-1));
rmse_pers_test = rmse(sn(ind_test), sn(ind_test-1));
%% figure: sunspot number series
figure();
LineWidth = 3;
FontSize = 30;

plot(epoch(1:length(sn)),sn,'k','LineWidth',LineWidth); hold on
plot(epoch(ind_train),train,'b','LineWidth',LineWidth); hold on
plot(epoch(ind_val),val,'Color','#D95319','LineWidth',LineWidth); hold on
plot(epoch(ind_test),test,'g','LineWidth',LineWidth); hold on
plot(epoch(length(sn)+1:length(sn)+future_step)+dev/12-6/12,future,'r','LineWidth',LineWidth); hold on
% plot(epoch(length(sn)+1:length(sn)+future_step),future_ulim,':','Color','#7E2F8E','LineWidth',LineWidth); hold on
% plot(epoch(length(sn)+1:length(sn)+future_step),future_llim,':','Color','#A2142F','LineWidth',LineWidth); 
grid on
legend('ISN','train','valid','test','predict')
xlim([epoch(1),epoch(end)+5])
ylim([0,250])
xlabel('Year')
ylabel('SSN')
set(gca,'LineWidth',LineWidth/2,'FontSize',FontSize)
%% functions
function rmse = rmse(observation, prediction)
    rmse = sqrt(mean((prediction-observation).^2));
end
