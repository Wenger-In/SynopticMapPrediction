clear; close all;
cfg = project_config();
save_or_not = 1;
cr_beg = 2259;
cr_end = 2271;
for i_cr = cr_beg : cr_end
    %% STEP 1: import from original format
    store_dir = [cfg.paths.wso_download_dat, filesep];
    file_name = ['CR',num2str(i_cr),'.dat'];
    data_dir = [store_dir,file_name];
    data_cell = textread(data_dir,'%s','delimiter','\n');
    %% extract gs and hs
    gs = zeros(10,10);
    hs = zeros(10,10);
    for i_row = [1:10,12:21]
        data_cell_sub = data_cell(i_row);
        data_str_sub = cell2mat(data_cell_sub);
        if i_row == 2
            data_sub = str2num(data_str_sub(1:18));
        elseif i_row == 13
            data_sub = str2num(data_str_sub(1:18));
        else
            data_sub = str2num(data_str_sub);
        end
        if i_row < 11
            gs(i_row,1:length(data_sub)-1) = data_sub(2:end);
        else
            hs(i_row-11,1:length(data_sub)-1) = data_sub(2:end);
        end
    end
    %% STEP 2: save as readable format
    l_max = 9;
    row_max = (1 + l_max+1)*(l_max+1)/2;
    %% construct l_lst and m_lst
    l_lst = [];
    m_lst = [];
    for i_l = 0 : l_max
        l_lst_sub = zeros(1,i_l+1) + i_l;
        l_lst = [l_lst;l_lst_sub.'];
        m_lst_sub = 0 : i_l;
        m_lst = [m_lst;m_lst_sub.'];
    end
    %% put in coefficients
    coef = zeros(row_max,2);
    for i_row = 1 : row_max
        coef(i_row,1) = gs(l_lst(i_row) + 1,m_lst(i_row) + 1);
        coef(i_row,2) = hs(l_lst(i_row) + 1,m_lst(i_row) + 1);
    end
    %% save data
    data = [l_lst,m_lst,coef];
    save_dir = [cfg.paths.wso_harmonics, filesep];
    save_file = [save_dir,'cr',num2str(i_cr),'.dat'];
    if save_or_not == 1
        save(save_file,'data','-ascii');
    end
end
