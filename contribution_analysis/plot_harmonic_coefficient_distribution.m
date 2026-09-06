clear; close all;
cfg = project_config();
save_or_not = 0;
%% Component: colorbar red-white-blue
color_red   = [1,0,0];
color_white = [1,1,1];
color_blue  = [0,0,1];
n1 = 100;
n2 = 100;
R_comp = [linspace(color_red(1),color_white(1),n1),linspace(color_white(1),color_blue(1),n2)];
G_comp = [linspace(color_red(2),color_white(2),n1),linspace(color_white(2),color_blue(2),n2)];
B_comp = [linspace(color_red(3),color_white(3),n1),linspace(color_white(3),color_blue(3),n2)];
red_white_blue = [R_comp',G_comp',B_comp'];
%% Component


%% import data
store_dir = [cfg.paths.wso_root, filesep];
file_name = 'gather_harmonic_coefficient.dat';
data_dir = [store_dir,file_name];
data = load(data_dir);
% data profile
l_lst = data(1,:);
m_lst = data(2,:);
hc_mat = data(3:end,:);
%% construct grid
cr_beg_WSO = 1642; cr_end_WSO = 2258;
cr_lst = cr_beg_WSO : cr_end_WSO;
cr_num = length(cr_lst);
l_arr = min(l_lst) : 1 : max(l_lst)+1;
m_arr = max(m_lst) :-1 : min(m_lst)-1;
[ll,mm] = meshgrid(l_arr,m_arr);
%% extract data 
for i_cr = 1 : cr_num
    close all;
    cr_sub = cr_lst(i_cr);
    hc_sub = hc_mat(i_cr,:);
    hhcc = zeros(length(m_arr),length(l_arr));
    m_base = 10;
    l_base = 1;
    for i_col = 1 : length(hc_sub)
        l_col = l_lst(i_col);
        m_col = m_lst(i_col);
        hhcc(m_base-m_col,l_base+l_col) = hc_sub(i_col);
    end
    hhcc(hhcc==0) = nan;
    %% plot figure
    figure();
    LineWidth = 2;
    FontSize = 15;
    hp = pcolor(ll,mm,hhcc);
    clim = max(max(abs(hhcc)));
    colormap(red_white_blue); colorbar;
    hold on
    for i_l = 1 : length(l_arr)-1
        for i_m = 1 : length(m_arr)-1
            text(l_base+i_l-1,m_base-i_m-0.5,num2str(roundn(hhcc(i_m,i_l),-1)),'HorizontalAlignment','right','Color','k','FontSize',10);
        end
    end
    xlabel('order (l)')
    ylabel('degree (m)')
    title(['CR',num2str(cr_sub)])
    whitebg('k');
    set(gca,'Clim',[-clim,clim],'LineWidth',LineWidth,'FontSize',FontSize);
    %% save figure
save_dir = [fullfile(cfg.output_root, 'hc_distribution'), filesep];
    save_name = ['CR',num2str(cr_sub)];
    if save_or_not == 1
        exportgraphics(gca,[save_dir,save_name,'.png'],'BackgroundColor','current')
    end
end
