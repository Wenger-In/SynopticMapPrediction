clear; close all;
cfg = project_config();
save_or_not = 0;
mode = 1; % 0-GONG, 1-WSO, 2-predict
cr_beg= 1700;
cr_end= 1700;
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
%% import data
if mode == 0 % for GONG: 2047-2268
    store_dir = [cfg.paths.gong_harmonics, filesep];
elseif mode == 1 % for WSO: 1642-2272
    store_dir = [cfg.paths.wso_harmonics, filesep];
elseif mode == 2 % for predict: 2259-2408
    store_dir = [fullfile(cfg.output_root, 'predict_SC25', 'harmonics'), filesep];
end
save_dir = [fullfile(cfg.output_root, 'harmonics_map'), filesep];
for cr = cr_beg : cr_end
    close all;
    if mode == 0 % for GONG
        file_name = ['mrmqc_c',num2str(cr),'.dat'];
    else % for WSO
        file_name = ['cr',num2str(cr),'.dat'];
    end
    data_dir = [store_dir,file_name];
    data = load(data_dir);
    g = data(:,3);% coeff of cos
    h = data(:,4);% coeff of sin
    %% extract grid
    theta = linspace(0,pi,180); % co-latitude [rad.]
    phi = linspace(0,2*pi,360); % longitude [rad.]
    lon = phi./pi.*180;
    lat = (pi/2 - theta)./pi.*180;
    ord_max = 60; % max value of spherical order
    %% select order and level
    l_lst = 0:5;
    m_lst = -5:5;
    %% spherical harmonic functions
    magneto = zeros(180,360);
    for i_l = 0 : ord_max % order of harmonics
        if ismember(i_l,l_lst)
            % modified from 'sch' option of matlab legendre function
            P = legendre(i_l,cos(theta),'sch'); % /sqrt(2*pi);
            temp = (i_l + 1)*i_l/2 + 1;
            for i_m = 1 : i_l + 1 % degree of harmonics
                i_row = temp + i_m - 1;
                renorm = 1; % already normalized
                %             renorm = sqrt(2*i_ord+1); % un-normalized
                m = i_m - 1;
                if ismember(m,m_lst)
                    triang = g(i_row)*cos(m*phi);
                    magneto = magneto + renorm*P(i_m,:).'*triang;
                end
                if ismember(-m,m_lst)
                    triang = h(i_row)*sin(m*phi);
                    magneto = magneto + renorm*P(i_m,:).'*triang;
                end
            end
        end
    end
    if mode ~= 0
        magneto = magneto / 100; % [G]
    end
    %% plot magnetogram in plane
    figure();
    LineWidth = 2;
    FontSize = 15;
    clim = max(max(abs(magneto)));
%     clim = 300;
%     clim = inf;

    hp = pcolor(lon,lat,magneto);
    set(hp,'LineStyle','none');
    cb = colorbar;
    cb.Title.String = '[G]';
    colormap(red_white_blue);
    hold on
%     yline(0,'--k','LineWidth',LineWidth);
    axis equal
    xlim([0 360]);
    ylim([-90 90]);
    xticks([0 90 180 270 360]);
    xticklabels({'0^\circ','90^\circ','180^\circ','270^\circ','360^\circ'})
    yticks([-90 -45 0 45 90]);
    yticklabels({'-90^\circ','-45^\circ','0^\circ','45^\circ','90^\circ'})
    set(gca,'Clim',[-clim,clim],'TickDir','out','XminorTick','on','YminorTick','on','LineWidth',LineWidth,'FontSize',FontSize);

%     title(save_name,'FontSize',FontSize);
%     title(['CR ',num2str(cr)]);

    % save_name = ['l=',num2str(l_lst),',m=',num2str(m_lst)]; % for test
    if mode == 0 % for GONG
        save_name = ['GONG_map_harmonics_',num2str(cr)];
    elseif mode == 1% for WSO
        save_name = ['WSO_map_harmonics_',num2str(cr)];
    elseif mode == 2 % for predict
        save_name = ['WSO_pred_harmonics_',num2str(cr)];
    end
    if save_or_not == 1
        %     saveas(gca,[save_dir,'plane\',save_name,'_plane.png']); % for test
        saveas(gca,[save_dir,save_name,'.png']); % for GONG and WSO
    end
    %% plot magnetogram in sphere
    figure('color','k');
    
    [ttheta,pphi] = meshgrid(theta,phi);
    sphx = sin(ttheta).*cos(pphi);
    sphy = sin(ttheta).*sin(pphi);
    sphz = cos(ttheta);
    surf(sphx,sphy,sphz,magneto.');
    shading interp
    ax = gca;
    ax.Color = 'k';
    colorbar off
    colormap(red_white_blue);
    axis equal
    set(gca,'Clim',[-clim,clim],'GridColor','w','XminorTick','on','YminorTick','on','LineWidth',LineWidth,'FontSize',FontSize);
    grid off
    axis off
    box off
    view([-1,0,0])
    
    save_name = ['l=',num2str(l_lst),',m=',num2str(m_lst)];
    % title(save_name,'FontSize',FontSize);
%     title(['CR ',num2str(cr)]);
    if save_or_not == 1
        saveas(gca,[save_dir,'sphere\',save_name,'_sphere.png']);
    end
end
