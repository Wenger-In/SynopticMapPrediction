clear; close all;
cfg = project_config();
save_or_not = 0;
%% CR series
% cr_beg_GNG = 2047; cr_end_GNG = 2266; % for GONG data
cr_beg_WSO = 1642; cr_end_WSO = 2258; % for WSO data
cr_lst = cr_beg_WSO : cr_end_WSO;
cr_num = length(cr_lst);
%% coefficient series
max_order = 9; % maximum harmonics order
hc_num = (1+(2*max_order+1))*(max_order+1)/2;
hc_mat = zeros(cr_num,hc_num);
l_lst = zeros(1,hc_num);
m_lst = zeros(1,hc_num);
%% import data
store_dir = [cfg.paths.wso_harmonics, filesep];
for i = cr_beg_WSO : cr_end_WSO
%     file_name = ['mrmqc_c',num2str(i_cr),'.dat'];
    file_name = ['cr',num2str(i),'.dat'];
    data_dir = [store_dir,file_name];
    data = load(data_dir);
    % extract coefficients
    l_arr = data(:,1);
    m_arr = data(:,2);
    g_arr = data(:,3);
    h_arr = data(:,4);
    % sort to series
    i_cr = i - cr_beg_WSO + 1;
    i_col = 1;
    for i_l = 0 : max_order
        for i_m = 0 : i_l
            l_lst(i_col) = i_l;
            i_row = find(l_arr == i_l & m_arr == i_m);
            if i_m == 0
                hc_mat(i_cr,i_col) = g_arr(i_row);
            else
                hc_mat(i_cr,i_col) = g_arr(i_row);
                m_lst(i_col) = i_m;
                i_col = i_col + 1;
                l_lst(i_col) = i_l;
                hc_mat(i_cr,i_col) = h_arr(i_row);
                m_lst(i_col) = -i_m;
            end
            i_col = i_col + 1;
        end
    end
end
%% save data
save_dir = [cfg.paths.wso_root, filesep];
save_var = [l_lst;m_lst;hc_mat];
if save_or_not == 1
    save_file = [save_dir,'harmonic_coefficient_gathered.dat'];
    save(save_file,'save_var','-ascii');
    disp('Neaten WSO Harmonic Coefficients Successfully');
end
