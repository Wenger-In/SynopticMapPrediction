clear; close all;
cfg = project_config();
save_or_not = 0;
%% work dir
WSO_dir = [cfg.paths.wso_field, filesep];
save_dir = [fullfile(cfg.output_root, 'WSO_map'), filesep];
cr_beg = 2239;
cr_end = 2239;
for i_cr = cr_beg : cr_end
    close all;
    %% import data
    file_name = ['cr',num2str(i_cr),'.dat'];
    file_dir = [WSO_dir,file_name];
    Br = importdata(file_dir); % [uT]
    Br = Br/100; % [G]
    lon_dir = [WSO_dir,'lon_arr.dat'];
    lon = importdata(lon_dir); % [deg.]
    lat_dir = [WSO_dir,'lat_arr.dat'];
    lat = importdata(lat_dir); % [deg.]
    %% colorbar red-white-blue
    color_red   = [1,0,0];
    color_white = [1,1,1];
    color_blue  = [0,0,1];
    n1 = 100;
    n2 = 100;
    R_comp = [linspace(color_red(1),color_white(1),n1),linspace(color_white(1),color_blue(1),n2)];
    G_comp = [linspace(color_red(2),color_white(2),n1),linspace(color_white(2),color_blue(2),n2)];
    B_comp = [linspace(color_red(3),color_white(3),n1),linspace(color_white(3),color_blue(3),n2)];
    red_white_blue = [R_comp',G_comp',B_comp'];
    %% plot GONG magnetogram
    figure();
    LineWidth = 2;
    FontSize = 15;

    h = pcolor(lon,lat,Br);
    set(h,'LineStyle','none');
    shading interp
    cb = colorbar;
    cb.Title.String = '[G]';
    colormap(red_white_blue);
    hold on
    yline(0,'--k','LineWidth',LineWidth);
    axis equal
    xlim([0 360]);
    ylim([-90 90]);
    xticks([0 90 180 270 360]);
    xticklabels({'0^\circ','90^\circ','180^\circ','270^\circ','360^\circ'})
    yticks([-90 -45 0 45 90]);
    yticklabels({'-90^\circ','-45^\circ','0^\circ','45^\circ','90^\circ'})
    clim = max(max(abs(Br)));
    set(gca,'CLim',[-clim clim],'TickDir','out','XminorTick','on','YminorTick','on','LineWidth',LineWidth,'FontSize',FontSize);

    save_name = ['WSO_map_',num2str(i_cr)];
%     title(['WSO map (CR',num2str(i_cr),')'],'FontSize',FontSize);
    if save_or_not == 1
        saveas(gca,[save_dir,save_name,'.png']);
    end
end
