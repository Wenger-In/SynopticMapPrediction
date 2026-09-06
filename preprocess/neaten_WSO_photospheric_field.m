clear; close all;
cfg = project_config();
save_or_not = 0;
% cr_beg = 1642;
cr_beg = 2272; %%%%% there is no data for CR2208 in WSO's web! %%%%%
cr_end = 2287;
for i_cr = cr_beg : cr_end
    %% STEP 1: import from original format
    store_dir = [cfg.paths.wso_download_txt, filesep];
    file_name = ['CR',num2str(i_cr),'.txt'];
    data_dir = [store_dir,file_name];
    data_cell = textread(data_dir,'%s','delimiter','\n');
    %% construct grid
    lon = linspace(360,0,73); % [deg.]
    lat_sin = linspace(14.5/15,-14.5/15,30);
    lat = asind(lat_sin); % [deg.]
    Br = zeros(length(lat),length(lon));
    %% extract magnetic field
    info_cell = data_cell(2);
    info = cell2mat(info_cell);
    for i_lon = 1 : length(lon)
        i_sub = 4 * i_lon;
        % first row of each longtitude
        data_cell_1 = data_cell(i_sub);
        data_str_1 = cell2mat(data_cell_1);
        data_sub = s2n(data_str_1(11:end));
        % following rows of each longitude
        for i_234 = 2 : 4
            data_cell_234 = data_cell(i_sub+i_234-1);
            data_str_234 = cell2mat(data_cell_234);
            data_234 = s2n(data_str_234);
            data_sub = [data_sub,data_234];
        end
        Br(:,i_lon) = data_sub.';
    end
    %% STEP 2: save as readable format
    save_dir = [cfg.paths.wso_field, filesep];
    save_file = [save_dir,'cr',num2str(i_cr),'.dat'];
    if save_or_not == 1
        save(save_file,'Br','-ascii');
        save([save_dir,'lon_arr.dat'],'lon','-ascii');
        save([save_dir,'lat_arr.dat'],'lat','-ascii');
        disp(['Neaten CR',num2str(i_cr),'.dat Successfully']);
    end
end



%% function
function num = s2n(str)
    if ~contains(str,'-')
        num = str2num(str);
    else
        spl = split(str,'-');
        sz = size(spl);
        spl_1 = cell2mat(spl(1));
        num = str2num(spl_1);
        for i = 2 : sz
            spl_234 = cell2mat(spl(i));
            num_234 = str2num(spl_234);
            num_234(1) = -num_234(1);
            num = [num,num_234];
        end
    end
end
