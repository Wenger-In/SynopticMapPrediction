clear; close all;
cfg = project_config();
save_or_not = 0;
%% data path
pred_dir = [fullfile(cfg.output_root, 'harmonics_map', 'WSO'), filesep];
WSO_dir = [fullfile(cfg.repo_root, 'determine_order'), filesep];
%% interpolate
cr_beg = 2239;
cr_end = 2239;
for cr = cr_beg : cr_end
    %% GONG grid
    lon_gong = linspace(0,360,360);
    lat_sin_gong = linspace(-1,1,180);
    lat_gong = asind(lat_sin_gong);
    [llon_gong,llat_gong] = meshgrid(lon_gong,lat_gong);
    %% WSO
    % WSO field
    WSO_path = [num2str(cr),'_WSO.mat'];
    WSO_Br = load([WSO_dir, WSO_path]);
    WSO_Br = WSO_Br.Br; % [G]
    % WSO grid
    WSO_grid_dir = [cfg.paths.wso_field, filesep];
    lon_dir = [WSO_grid_dir,'lon_arr.dat'];
    lon_WSO = importdata(lon_dir); % [deg.]
    lat_dir = [WSO_grid_dir,'lat_arr.dat'];
    lat_WSO = importdata(lat_dir); % [deg.]
    [llon_WSO, llat_WSO] = meshgrid(lon_WSO, lat_WSO);
    % interpolate predicted field into gong grid
    pred_Br_interp_WSO = interp2(llon_WSO, llat_WSO, WSO_Br, llon_gong, llat_gong, 'linear');
    %% predict
    % predict field
%     pred_name = ['cr',num2str(cr),'_obs.mat'];
%     pred_Br = load([pred_dir,pred_name]); % [uT]
    pred_name = [num2str(cr), '_WSO_5.mat'];
    pred_Br = load([WSO_dir, pred_name]);
    pred_Br = pred_Br.magneto; % [G]
    pred_Br = flipud(pred_Br);
    % predicted grid
    lon_pred = linspace(0,360,360);
    lat_pred = linspace(-90,90,180);
    [llon_pred,llat_pred] = meshgrid(lon_pred,lat_pred);
    % interpolate predicted field into gong grid
    pred_Br_interp = interp2(llon_pred, llat_pred, pred_Br, llon_gong, llat_gong, 'linear');
    % save as .mat file
    if save_or_not == 1
    save_dir = [fullfile(cfg.repo_root, 'postprocess_on_class'), filesep];
        save_name = ['cr',num2str(cr),'_interp.mat'];
        save([save_dir,save_name],'pred_Br_interp')
    end
end
