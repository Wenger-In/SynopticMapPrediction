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
%% Compoent


%% Method 1: Use NOAA numbered Active Region
% %% PART 1: Import data
% % data profile: CR; No.; lat; lon_obs; lon; lat_ext; lon_ext
% data = data(data(:,1)~=0,:);
% %% PART 1
% 
% 
% %% PART 2: Select data
% cr_lst = min(data(:,1)) : max(data(:,1));
% for i_cr = 1 : length(cr_lst)
%     close all;
%     cr_sub = cr_lst(i_cr);
%     data_sub = data(data(:,1)==cr_sub,:);
%     No_sub = unique(data_sub(:,2));
%     data_slc = [];
%     x_rect = [];
%     y_rect = [];
%     w_rect = [];
%     h_rect = [];
%     for i_No = 1 : length(No_sub)
%         No_subb = No_sub(i_No);
%         i_sub = find(data_sub(:,2)==No_subb);
%         data_subb = data_sub(i_sub,:);
%         data_slc_sub = select_nearest(data_subb);
%         data_slc = [data_slc;data_slc_sub];
%         x_rect = [x_rect; data_slc_sub(5) - data_slc_sub(7)/2];
%         y_rect = [y_rect; data_slc_sub(3) - data_slc_sub(6)/2];
%         w_rect = [w_rect; data_slc_sub(7)];
%         h_rect = [h_rect; data_slc_sub(6)];
%     end
%     %% PART 2
%     %% PART 3: Plot marked magnetogram
%     [lon,lat,Br] = construct_field(cr_sub);
%     % plot figure
%     figure();
%     LineWidth = 2;
%     FontSize = 15;
%     % plot magnetogram
%     hp = pcolor(lon,lat,Br);
%     shading interp
%     set(hp,'LineStyle','none');
%     colorbar;
%     colormap(red_white_blue);
%     % plot rectangle
%     hold on
%     for i_rect = 1 : length(x_rect)
%         rectangle('Position',[x_rect(i_rect),y_rect(i_rect),w_rect(i_rect),h_rect(i_rect)], ...
%             'EdgeColor','k','LineWidth',LineWidth);
%     end
%     % set figure properties
%     axis equal
%     xlim([0 360]);
%     ylim([-90 90]);
%     title(['CR',num2str(cr_sub)],'FontSize',FontSize);
%     clim = max(max(abs(Br)));
%     set(gca,'Clim',[-clim,clim],'TickDir','out','XminorTick','on','YminorTick','on','LineWidth',LineWidth,'FontSize',FontSize);
%     % save figure
%     save_name = ['CR',num2str(cr_sub)];
%     if save_or_not == 1
%         saveas(gca,[save_dir,save_name,'.png']);
%     end
%     %%
% end
%% Method 1


%% Method 2: Select Active Region manually
%% Part 1: Import data
% import WSO field data
WSO_dir = [cfg.paths.wso_field, filesep];
cr = 1904;%1970
WSO_name = ['cr',num2str(cr),'.dat'];
WSO_file = [WSO_dir,WSO_name];
Br = importdata(WSO_file); % [uT]
lon_dir = [WSO_dir,'lon_arr.dat'];
lon_WSO = importdata(lon_dir); % [deg.]
lat_dir = [WSO_dir,'lat_arr.dat'];
lat_WSO = importdata(lat_dir); % [deg.]
[lonn,latt] = meshgrid(lon_WSO,lat_WSO);
% import WSO harmonic coefficients data
hc_dir = [cfg.paths.wso_root, filesep];
hc_file = 'gather_harmonic_coefficient.mat';
hc_save = load([hc_dir,hc_file]);
hc_data =  hc_save.save_var;
l_lst = hc_data(1,:); l_arr = min(l_lst) : 1 : max(l_lst)+1;
m_lst = hc_data(2,:); m_arr = max(m_lst) :-1 : min(m_lst)-1;
[ll,mm] = meshgrid(l_arr,m_arr);
hc_mat = hc_data(3:end,:);
% CR list data
cr_beg = 1642;
cr_end = 2258;
cr_lst = cr_beg : cr_end;
%% PART 1


%% PART 2: Select Active Region
% plot WSO magnetogram
figure();
LineWidth = 2;
FontSize = 15;

h = pcolor(lon_WSO,lat_WSO,Br);
set(h,'LineStyle','none');
colorbar;
colormap(red_white_blue);
axis equal
xlim([0 360]);
ylim([-90 90]);
clim = max(max(abs(Br)));
title(['CR ',num2str(cr)]);
set(gca,'CLim',[-clim clim],'TickDir','out','XminorTick','on','YminorTick','on','LineWidth',LineWidth,'FontSize',FontSize);

% selected regions
Br_cut = Br;
Br_cut([1:5,26:30],:) = 0;
Br_cut(abs(Br_cut)<150) = 0; Br_cut(:,[1,20]) = 0;

figure();
LineWidth = 2;
FontSize = 15;

h = pcolor(lon_WSO,lat_WSO,Br_cut);
set(h,'LineStyle','none');
colorbar;
colormap(red_white_blue);
axis equal
xlim([0 360]);
ylim([-90 90]);
clim = max(max(abs(Br)));
title(['CR ',num2str(cr)]);
set(gca,'CLim',[-clim clim],'TickDir','out','XminorTick','on','YminorTick','on','LineWidth',LineWidth,'FontSize',FontSize);
%% PART 2


%% PART 3: Calculate contribution of harmonic coefficients
l_max = 9;
hc_num = (1+2*l_max+1)*(l_max+1)/2 - 1;
% construct Br contribution
Br_cont_interp = zeros(length(lat_WSO),length(lon_WSO),hc_num);
l_cont = zeros(hc_num);
m_cont = zeros(hc_num);
% calculate
for i_l = 1 : l_max
    for i_m = -i_l : i_l
        i_hc = i_l^2+i_m+i_l;
        [lon,lat,Br_cont] = get_harmonics_contribution(cr,i_l,i_m);
        Br_cont_interp(:,:,i_hc) = griddata(lon,lat,Br_cont,lonn,latt);
        l_cont(i_hc) = i_l;
        m_cont(i_hc) = i_m;
    end
end
%% PART 3

%% PART 4: Classification
m_base = 10;
l_base = 1;
% north postive region
[nor_pos_row,nor_pos_col] = find(Br_cut(1:15,:)>0);
[nor_pos_cont,nor_pos_sum,nor_pos_tot] = cal_class_contribution(l_arr,m_arr,nor_pos_row,nor_pos_col,Br_cut,Br_cont_interp,l_cont,m_cont);
% north negative region
[nor_neg_row,nor_neg_col] = find(Br_cut(1:15,:)<0);
[nor_neg_cont,nor_neg_sum,nor_neg_tot] = cal_class_contribution(l_arr,m_arr,nor_neg_row,nor_neg_col,Br_cut,Br_cont_interp,l_cont,m_cont);
% south postive region
[sou_pos_row,sou_pos_col] = find(Br_cut(16:end,:)>0);
[sou_pos_cont,sou_pos_sum,sou_pos_tot] = cal_class_contribution(l_arr,m_arr,15+sou_pos_row,sou_pos_col,Br_cut,Br_cont_interp,l_cont,m_cont);
% south negative region
[sou_neg_row,sou_neg_col] = find(Br_cut(16:end,:)<0);
[sou_neg_cont,sou_neg_sum,sou_neg_tot] = cal_class_contribution(l_arr,m_arr,15+sou_neg_row,sou_neg_col,Br_cut,Br_cont_interp,l_cont,m_cont);
%% PART 4


%% PART 5: Plot contribution
figure();
subplot_class_contribution(2,2,1,l_arr,m_arr,nor_pos_cont,nor_pos_sum,nor_pos_tot,'North Postive Active Region',red_white_blue)
subplot_class_contribution(2,2,2,l_arr,m_arr,nor_neg_cont,nor_neg_sum,nor_neg_tot,'North Negative Active Region',red_white_blue)
subplot_class_contribution(2,2,3,l_arr,m_arr,sou_pos_cont,sou_pos_sum,sou_pos_tot,'South Postive Active Region',red_white_blue)
subplot_class_contribution(2,2,4,l_arr,m_arr,sou_neg_cont,sou_neg_sum,sou_neg_tot,'South Negative Active Region',red_white_blue)
%% PART 5
% whitebg('w');
%% Method 2


%% functions
function data_select = select_nearest(data_subb)
    lon_obs_lst = data_subb(:,4);
    lon_obs_lst(lon_obs_lst>180) = 360 - lon_obs_lst(lon_obs_lst>180);
    i_select_lst = find(lon_obs_lst==min(lon_obs_lst));
    i_select = i_select_lst(1);
    data_select = data_subb(i_select,:);
end

function [lon,lat,Br] = construct_field(cr)
WSO_dir = [cfg.paths.wso_field, filesep];
    file_name = ['cr',num2str(cr),'.dat'];
    file_dir = [WSO_dir,file_name];
    Br = importdata(file_dir); % [uT]
    lon_dir = [WSO_dir,'lon_arr.dat'];
    lon = importdata(lon_dir); % [deg.]
    lat_dir = [WSO_dir,'lat_arr.dat'];
    lat = importdata(lat_dir); % [deg.]
end

function [lon,lat,field_contrib] = get_harmonics_contribution(cr,l,m)
    % import data
store_dir = [cfg.paths.wso_harmonics, filesep];
    file_name = ['cr',num2str(cr),'.dat'];
    data_dir = [store_dir,file_name];
    data = load(data_dir);
    l_lst = data(:,1);
    m_lst = data(:,2);
    g_lst = data(:,3);% coeff of cos
    h_lst = data(:,4);% coeff of sin
    % extract grid
    theta = linspace(0,pi,180); % co-latitude [rad.]
    phi = linspace(0,2*pi,360); % longitude [rad.]
    lon = phi./pi.*180;
    lat = (pi/2 - theta)./pi.*180;
    % spherical harmonic contribution
    i_row = find(l_lst==l & m_lst==abs(m));
    % modified from 'sch' option of matlab legendre function
    P = legendre(l,cos(theta),'sch'); % /sqrt(2*pi);
    renorm = 1; % already normalized
    if m >= 0
        triang = g_lst(i_row)*cos(m*phi);
        field_contrib = renorm*P(m+1,:).'*triang;
    elseif m < 0
        triang = h_lst(i_row)*sin(-m*phi);
        field_contrib = renorm*P(-m+1,:).'*triang;
    end
end

function [class_contrib,class_sum,class_total] = cal_class_contribution(l_arr,m_arr,row_lst,col_lst,Br_cut,Br_cont_interp,l_cont,m_cont)
    class_contrib = zeros(length(m_arr),length(l_arr));
    class_total = 0;
    l_max = 9;
    m_base = 10;
    l_base = 1;
    for i_class = 1 : length(row_lst)
        for i_l = 1 : l_max
            for i_m = -i_l : i_l
                i_lm = find(l_cont==i_l & m_cont==i_m);
                class_contrib(m_base-i_m,l_base+i_l) = class_contrib(m_base-i_m,l_base+i_l) ...
                    + Br_cont_interp(row_lst(i_class),col_lst(i_class),i_lm);
            end
        end
        class_total = class_total + Br_cut(row_lst(i_class),col_lst(i_class));
    end
    class_sum = sum(sum(class_contrib));
    class_contrib(class_contrib==0) = nan;
end

function subplot_class_contribution(m,n,p,l_arr,m_arr,class_contrib,class_sum,class_tot,class_title,red_white_blue)
    [ll,mm] = meshgrid(l_arr,m_arr);
    LineWidth = 2;
    FontSize = 15;
    m_base = 10;
    l_base = 1;
    subplot(m,n,p)
    pcolor(ll,mm,class_contrib);
    clim = max(max(abs(class_contrib)));
    colormap(red_white_blue); colorbar;
    hold on
    for i_l = 1 : length(l_arr)-1
        for i_m = 1 : length(m_arr)-1
            text(l_base+i_l-1,m_base-i_m-0.5,num2str(roundn(class_contrib(i_m,i_l),-1)),'HorizontalAlignment','right','Color','k','FontSize',10);
        end
    end
    xlabel('order (l)')
    ylabel('degree (m)')
    title([class_title,' ',num2str(class_sum),'/',num2str(class_tot)])
    whitebg('k');
    set(gca,'Clim',[-clim,clim],'LineWidth',LineWidth,'FontSize',FontSize);
end
