clear; close all;
cfg = project_config();
l = 5;
cr_beg_WSO = 1642; cr_end_WSO = 2258;
cr_obs_lst = cr_beg_WSO : cr_end_WSO;

% observation
store_dir = [cfg.paths.wso_root, filesep];
file_name = 'gather_harmonic_coefficient.dat';
data_dir = [store_dir,file_name];
data = load(data_dir);

% extract data
l_lst = data(1,:);
m_lst = data(2,:);
hc_mat = data(3:end,:);

% select data
m_num = 2*l + 1;
col_beg = l^2 + 1;
col_end = (l + 1)^2;
m_sub = m_lst(col_beg:col_end);
hc_sub = hc_mat(:,col_beg:col_end);

% figure properties
LineWidth = 1.5;
FontSize = 14;
h_sub_lst = [0.2,0.15,0.12,0.09,0.07,0.061,0.055,0.048,0.042];
h_space_lst = [0.05,0.02,0.01,0.01,0.01,0.008,0.005,0.005,0.005];

fig_num = 5;
for i_m = 1 : m_num
    % prediction data
pred_dir = [fullfile(cfg.output_root, 'predict', 'model_output'), filesep];
    pred = importdata([pred_dir,'No_',num2str(col_beg+i_m-2),'.csv']);
    future_step = 150;
    cr_pred_lst = cr_end_WSO + future_step - length(pred) + 1 : cr_end_WSO + future_step;
    % plot figure
    i_fig = mod(i_m,fig_num);
    if i_fig == 1 % first panel, start a new figure
        figure();
        set(gcf,'Position',[0,0,600,800])
    end
    if i_fig == 0 % bottom panel, add the xtick
        subplot_hc(cr_obs_lst,hc_sub(:,i_m),0,h_sub_lst(2),h_space_lst(2),LineWidth,FontSize,['g_{',num2str(l),'}^{',num2str(m_sub(i_m)),'}']);
        plot(cr_pred_lst,pred,'r','LineWidth',LineWidth)
        xlim([min(cr_obs_lst),max(cr_pred_lst)])
        if i_m == m_num
            subplot_xtick()
        end
    else
        subplot_hc(cr_obs_lst,hc_sub(:,i_m),fig_num-i_fig,h_sub_lst(2),h_space_lst(2),LineWidth,FontSize,['g_{',num2str(l),'}^{',num2str(m_sub(i_m)),'}']);
        plot(cr_pred_lst,pred,'r','LineWidth',LineWidth)
        xlim([min(cr_obs_lst),max(cr_pred_lst)])
        if i_m == m_num
            subplot_xtick()
        end
    end
end


function subplot_hc(x,y,i_h,h_sub,h_space,LineWidth,FontSize,label)
    w_base = 0.1;
    w_sub = 0.8;
    h_base = 0.1;

    subplot('Position',[w_base,h_base+(h_sub+h_space)*i_h,w_sub,h_sub])
    plot(x,y,'k','LineWidth',LineWidth);
    hold on; grid on
    yline(0,'r','LineWidth',LineWidth,'Alpha',0.2)
    hold on
%     ylabel(label)
    set(gca,'LineWidth',LineWidth,'FontSize',FontSize,'XTickLabel',[],'XMinorTick','on')
end

function subplot_xtick()
    set(gca,'XTick',[1645,1700,1800,1900,2000,2100,2200,2300,2400], ...
        'XTickLabel',{'CR','1700','1800','1900','2000','2100','2200','2300','2400'}); % CR ticks
end
