clear; close all;
cfg = project_config();
cr_beg_WSO = 1642; cr_end_WSO = 2258;
cr_obs_lst = cr_beg_WSO : cr_end_WSO;

% observation
store_dir = [cfg.paths.wso_root, filesep];
file_name = 'gather_harmonic_coefficient.dat';
data_dir = [store_dir,file_name];
data = load(data_dir);

pred_dir = [fullfile(cfg.output_root, 'predict_SC25', 'model_output'), filesep];
pred = importdata([pred_dir,'No_1.csv']);
future_step = 150;
cr_pred_lst = cr_end_WSO + future_step - length(pred) + 1 : cr_end_WSO + future_step;

% extract data
hc_sub = data(3:end,2);
lb_lst = [25, 97, 150, 120];

% plot figure
LineWidth = 2;
FontSize = 25;
subplot_hc(cr_obs_lst,hc_sub,4,0.15,0.02,LineWidth,FontSize,'g_1^0','k');
plot(cr_pred_lst,pred,'r','LineWidth',LineWidth)
legend('observation','model output','Location','eastoutside')
xlim([min(cr_obs_lst),max(cr_obs_lst)+future_step])
for i = 1 : 4
    comp_imf = importdata([pred_dir,num2str(i-1),'_imf.csv']);
    subplot_hc(cr_obs_lst,comp_imf,4-i,0.15,0.02,LineWidth,FontSize,'','k');
    yline(0,'r','LineWidth',LineWidth,'Alpha',0.2)

    comp_train = importdata([pred_dir,num2str(i-1),'_train.csv']);
    train_lst = 1:length(comp_train);
    subplot_hc(train_lst+lb_lst(i)+1641,comp_train,4-i,0.15,0.02,LineWidth,FontSize,'','b');

    comp_val = importdata([pred_dir,num2str(i-1),'_val.csv']);
    val_lst = 1:length(comp_val);
    subplot_hc(val_lst+lb_lst(i)+1641+length(train_lst),comp_val,4-i,0.15,0.02,LineWidth,FontSize,'','#7E2F8E');

    comp_test = importdata([pred_dir,num2str(i-1),'_test.csv']);
    test_lst = 1:length(comp_test);
    subplot_hc(test_lst+lb_lst(i)+1641+length(train_lst)+length(val_lst),comp_test,4-i,0.15,0.02,LineWidth,FontSize,'','g');

    comp_future = importdata([pred_dir,num2str(i-1),'_future.csv']);
    future_lst = 1:length(comp_future);
    subplot_hc(future_lst+lb_lst(i)+1641+length(train_lst)+length(val_lst)+length(test_lst),comp_future,4-i,0.15,0.02,LineWidth,FontSize,'IMF1','r');
    xlim([min(cr_obs_lst),max(cr_obs_lst)+future_step])
    legend('observation','','train prediction','val prediction','test prediction','future prediction','Location','eastoutside')
end
subplot_xtick()

figure()
subplot_hc(cr_obs_lst,hc_sub,0,0.8,0.02,LineWidth,FontSize,'g_1^0','k');
plot(cr_pred_lst,pred,'r','LineWidth',LineWidth)
yline(0,'r','LineWidth',LineWidth,'Alpha',0.2)
legend('observation','model output','','Location','eastoutside')
xlim([min(cr_obs_lst),max(cr_obs_lst)+future_step])
subplot_xtick()


function subplot_hc(x,y,i_h,h_sub,h_space,LineWidth,FontSize,label,color)
    w_base = 0.1;
    w_sub = 0.8;
    h_base = 0.1;

    subplot('Position',[w_base,h_base+(h_sub+h_space)*i_h,w_sub,h_sub])
    plot(x,y,'Color',color,'LineWidth',LineWidth);
    hold on; grid on
    hold on
    ylabel(label)
    set(gca,'LineWidth',LineWidth,'FontSize',FontSize,'XTickLabel',[],'XMinorTick','on')
end

function subplot_xtick()
    set(gca,'XTick',[1645,1700,1800,1900,2000,2100,2200,2300,2400], ...
        'XTickLabel',{'CR','1700','1800','1900','2000','2100','2200','2300','2400'}); % CR ticks
end
