clear; close all;
cfg = project_config();
save_or_not = 1;
%% import original data
real_dir = [cfg.paths.wso_root, filesep];
ori_file = [real_dir,'gather_harmonic_coefficient.mat'];
hc_ori = importdata(ori_file);
l_lst = hc_ori(1,:);
m_lst = hc_ori(2,:);
%% import predicted data
pred_dir = [fullfile(cfg.output_root, 'EMD+LSTM', 'predict'), filesep];
future_step = 150;
max_order = 5;
hc_num = (max_order+1)^2;
l_lst_cut = l_lst(1:hc_num);
m_lst_cut = m_lst(1:hc_num);
% construct empty matrix for save
hc_pred = zeros(future_step,hc_num);
% put in predicted data
for i_hc = 1 : hc_num
    l_sub = l_lst(i_hc);
    m_sub = m_lst(i_hc);
    pred_name = ['g_',num2str(l_sub),'_',num2str(m_sub),'.csv'];
    pred_file_sub = [pred_dir,pred_name];
    data_pred_sub = importdata(pred_file_sub);
    hc_pred(:,i_hc) = data_pred_sub;
end
%% organize as common format
eg_file = [real_dir,'\harmonics\cr1642.dat'];
data_eg = importdata(eg_file);
data_eg_cut = data_eg(1:(max_order+2)*(max_order+1)/2,:);
for i_cr = 1 : future_step
    hc_sub = hc_pred(i_cr,:);
    data_pred = data_eg_cut;
    l_lst_eg = data_pred(:,1);
    m_lst_eg = data_pred(:,2);
    for i_hc = 1 : hc_num
        i_row = find(l_lst_eg==l_lst(i_hc) & m_lst_eg==abs(m_lst(i_hc)));
        if m_lst(i_hc) >= 0
            data_pred(i_row,3) = hc_sub(i_hc);
        else
            data_pred(i_row,4) = hc_sub(i_hc);
        end
    end
    save_file = [pred_dir,'cr',num2str(2258+i_cr),'.dat'];
    if save_or_not == 1
        save(save_file,'data_pred','-ascii');
    end
end
