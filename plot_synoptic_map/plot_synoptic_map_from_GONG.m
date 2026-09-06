clear; close all;
cfg = project_config();
save_or_not = 0;
%% work dir
GONG_dir = [cfg.paths.gong_fits, filesep];
save_dir = [fullfile(cfg.output_root, 'GONG_map'), filesep];
cr_beg = 2049;
cr_end = 2049;
for i_cr = cr_beg : cr_end
    close all;
    %% import data
    file_name = ['mrzqs_c',num2str(i_cr),'.fits'];
    file_dir = [GONG_dir,file_name];
    data = fitsread(file_dir);
    lon = linspace(0,360,360);
    lat_sin = linspace(-1,1,180);
    lat = asin(lat_sin) * 180 / pi;
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

    h = pcolor(lon,lat,data*100); % convert from G to uT
    set(h,'LineStyle','none');
    cb = colorbar;
    colormap(red_white_blue);
    axis equal
    xlim([0 360]);
    ylim([-90 90]);
    set(gca,'CLim',[-500 500],'TickDir','out','XminorTick','on','YminorTick','on','LineWidth',LineWidth,'FontSize',FontSize);

    save_name = ['GONG_map_',num2str(i_cr)];
%     title(['GONG map (CR',num2str(i_cr),')'],'FontSize',FontSize);
    if save_or_not == 1
        saveas(gca,[save_dir,save_name,'.png']);
    end
end
