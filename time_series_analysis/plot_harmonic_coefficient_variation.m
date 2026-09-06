clear; close all;
cfg = project_config();
% select order
l = 1;
%% Component: colormap stand for Solar Cycle
cr_beg_WSO = 1642; cr_end_WSO = 2258;
cr_lst = cr_beg_WSO : cr_end_WSO;
cr_num = length(cr_lst);
cr_gap_1 = 1780; %1771
cr_gap_2 = 1915; 
cr_gap_3 = 2081; 
cr_gap_4 = 2225; %2222
n_1 = cr_gap_1 - cr_beg_WSO;
n_2 = cr_gap_2 - cr_gap_1;
n_3 = cr_gap_3 - cr_gap_2;
n_4 = cr_gap_4 - cr_gap_3;                 
n_5 = cr_end_WSO - cr_gap_4;
full_num = max([n_1,n_2,n_3,n_4,n_5]);
color_sc_1 = [1,0,0];
color_sc_2 = [0,1,0];
color_sc_3 = [0,0,1];
color_sc_4 = [1,0,1];
color_sc_5 = [0,0,0];
color_white = [1,1,1];
sc_full_1 = [linspace(color_sc_1(1),color_white(1),full_num); linspace(color_sc_1(2),color_white(2),full_num); linspace(color_sc_1(3),color_white(3),full_num)];
sc_full_2 = [linspace(color_sc_2(1),color_white(1),full_num); linspace(color_sc_2(2),color_white(2),full_num); linspace(color_sc_2(3),color_white(3),full_num)];
sc_full_3 = [linspace(color_sc_3(1),color_white(1),full_num); linspace(color_sc_3(2),color_white(2),full_num); linspace(color_sc_3(3),color_white(3),full_num)];
sc_full_4 = [linspace(color_sc_4(1),color_white(1),full_num); linspace(color_sc_4(2),color_white(2),full_num); linspace(color_sc_4(3),color_white(3),full_num)];
sc_full_5 = zeros(3,n_5) + color_sc_5.';
colormap_sc = [sc_full_1(:,1:n_1).'; sc_full_2(:,1:n_2).'; sc_full_3(:,1:n_3).'; sc_full_4(:,1:n_4).'; sc_full_5.'];
%% Component


%% PART 0: import data
store_dir = [cfg.paths.wso_root, filesep];
file_name = 'gather_harmonic_coefficient.dat';
data_dir = [store_dir,file_name];
data = load(data_dir);

% extract data
l_lst = data(1,:);
m_lst = data(2,:);
hc_mat = data(3:end,:);

% select data
m_num = 2*l + 1;
col_beg = l^2 + 1;
col_end = (l + 1)^2;
m_sub = m_lst(col_beg:col_end);
hc_sub = hc_mat(:,col_beg:col_end);

% figure properties
h_sub_lst = [0.2,0.15,0.12,0.09,0.07,0.061,0.055,0.048,0.042];
h_space_lst = [0.05,0.02,0.01,0.01,0.01,0.008,0.005,0.005,0.005];
% y_down_hc_lst = [-300,-350,-400,-800,-1000,-1200,-1200,-1500,-1500];
y_down_hc_lst = [-360,-350,-400,-850,-900,-1000,-1000,-1300,-1000];
y_down_ps_lst = [-12,-15,-18,-23,-27,-34,-40,-45,-58];
h_sub = h_sub_lst(l);
h_space = h_space_lst(l);
y_down_hc = y_down_hc_lst(l);
y_down_ps = y_down_ps_lst(l);
%% PART 0


%% PART 1: Time series
% plot figure: time series of harmonic coefficient (normalized)
LineWidth = 0.3;
FontSize = 24;%14
sz = 30;

% % style 1: all in one figure
% figure()
% for i_m = 1 : m_num
%     subplot_hc(cr_lst,hc_sub(:,i_m),m_num-i_m,h_sub,h_space,LineWidth,FontSize,sz,colormap_sc,['g_{',num2str(l),'}^{',num2str(m_sub(i_m)),'}']);
% end
% subplot_xtick(y_down_hc,FontSize)

% style 2: given-number subfigures per figure
fig_num = 5;
for i_m = 1 : m_num
    i_fig = mod(i_m,fig_num);
    if i_fig == 1 % first panel, start a new figure
        figure();
    end
    if i_fig == 0 % bottom panel, add the xtick
        subplot_hc(cr_lst,hc_sub(:,i_m),0,h_sub_lst(2),h_space_lst(2),LineWidth,FontSize,sz,colormap_sc,['{\itg}_{',num2str(l),'}^{',num2str(m_sub(i_m)),'}']);
        if i_m == m_num
            subplot_xtick(y_down_hc_lst(l),FontSize)
        end
    else
        subplot_hc(cr_lst,hc_sub(:,i_m),fig_num-i_fig,h_sub_lst(2),h_space_lst(2),LineWidth,FontSize,sz,colormap_sc,['{\itg}_{',num2str(l),'}^{',num2str(m_sub(i_m)),'}']);
        if i_m == m_num
            subplot_xtick(y_down_hc_lst(l),FontSize)
        end
    end
end

% calculate magnetic moment
% M_10 = calmoment(g_10,1);
% M_1p1 = calmoment(g_11,1); M_1n1 = calmoment(h_11,1);
% M_20 = calmoment(g_20,1);
% M_2p1 = calmoment(g_21,1); M_2n1 = calmoment(h_21,1);
% M_2p2 = calmoment(g_22,1); M_2n2 = calmoment(h_22,1);
% M_30 = calmoment(g_30,3);
% M_3p1 = calmoment(g_31,3); M_3n1 = calmoment(h_31,3);
% M_3p2 = calmoment(g_32,3); M_3n2 = calmoment(h_32,3);
% M_3p3 = calmoment(g_33,3); M_3n3 = calmoment(h_33,3);
% plot figure: time series of harmonic coefficient (unnormalized)
% figure()
% LineWidth = 2;
% FontSize = 14;
%
% subplot_hc(cr_lst,M_10, 9,LineWidth,FontSize,'M_{1}^{0}')
% subplot_hc(cr_lst,M_1p1,8,LineWidth,FontSize,'M_{1}^{1}')
% subplot_hc(cr_lst,M_1n1,7,LineWidth,FontSize,'M_{1}^{-1}')
% subplot_hc(cr_lst,M_30, 6,LineWidth,FontSize,'M_{3}^{0}')
% subplot_hc(cr_lst,M_3p1,5,LineWidth,FontSize,'M_{3}^{1}')
% subplot_hc(cr_lst,M_3n1,4,LineWidth,FontSize,'M_{3}^{-1}')
% subplot_hc(cr_lst,M_3p2,3,LineWidth,FontSize,'M_{3}^{2}')
% subplot_hc(cr_lst,M_3n2,2,LineWidth,FontSize,'M_{3}^{-2}')
% subplot_hc(cr_lst,M_3p3,1,LineWidth,FontSize,'M_{3}^{3}')
% subplot_hc(cr_lst,M_3n3,0,LineWidth,FontSize,'M_{3}^{-3}')
% subplot_xtick(FontSize)
%% PART 1


%% PART 2: Wavelet transformation: get cwt from Python code (get_harmonic_coefficient_cwt.py)
cr2year = 365.2422/27.2753; % from Carrington Rotation Period to Year
% import cwt data
cwt_dir = [fullfile(cfg.paths.cwt_output, 'cmor1.5-1.0_log'), filesep];
freq_name = 'freq.csv';
freq = importdata([cwt_dir,freq_name]);
freq_num = length(freq);
wave_mat = zeros(freq_num,cr_num,m_num);
for i_m = 1 : m_num
    cwt_name = ['cwt_',num2str(l),'^',num2str(m_sub(i_m)),'.csv'];
    wave_sub = importdata([cwt_dir,cwt_name]);
    for i_freq = 1 : freq_num
%         wave_mat(i_freq,:,i_m) = wave_sub(i_freq,:);
        wave_mat(i_freq,:,i_m) = str2num(wave_sub{i_freq,1});
    end
end
period_cr = 1./freq;
period_year = period_cr ./ cr2year;

% plot figure: power spectrum of harmonic coefficient
LineWidth = 0.3;
FontSize = 24;%14

% % style 1: all in one figure
% figure()
% for i_m = 1 : m_num - 1
%     subplot_ps(cr_lst,period_year,wave_mat(:,:,i_m),m_num-i_m,h_sub,h_space,LineWidth,FontSize,['g_{',num2str(l),'}^{',num2str(m_sub(i_m)),'}']);
%     subplot_ps_mean(period_year,wave_mat(:,:,i_m),m_num-i_m,h_sub,h_space,LineWidth,FontSize);
% end
% % bottom panel
% subplot_ps(cr_lst,period_year,wave_mat(:,:,m_num),0,h_sub,h_space,LineWidth,FontSize,['g_{',num2str(l),'}^{',num2str(m_sub(m_num)),'}']);
% subplot_xtick(y_down_ps,FontSize)
% subplot_ps_mean(period_year,wave_mat(:,:,m_num),0,h_sub,h_space,LineWidth,FontSize);

% style 2: given-number subfigures per figure
fig_num = 5;
for i_m = 1 : m_num
    i_fig = mod(i_m,fig_num);
    if i_fig == 1 % first panel, start a new figure
        figure();
    end
    if i_fig == 0 % bottom panel, add the xtick
        subplot_ps(cr_lst,period_year,wave_mat(:,:,i_m),0,h_sub_lst(2),h_space_lst(2),LineWidth,FontSize,['{\itg}_{',num2str(l),'}^{',num2str(m_sub(i_m)),'}']);
        if i_m == m_num
            subplot_xtick(y_down_ps_lst(2),FontSize)
        end
        subplot_ps_mean(period_year,wave_mat(:,:,i_m),0,h_sub_lst(2),h_space_lst(2),LineWidth,FontSize);
    else
        subplot_ps(cr_lst,period_year,wave_mat(:,:,i_m),fig_num-i_fig,h_sub_lst(2),h_space_lst(2),LineWidth,FontSize,['{\itg}_{',num2str(l),'}^{',num2str(m_sub(i_m)),'}']);
        if i_m == m_num
            subplot_xtick(y_down_ps_lst(2),FontSize)
        end
        subplot_ps_mean(period_year,wave_mat(:,:,i_m),fig_num-i_fig,h_sub_lst(2),h_space_lst(2),LineWidth,FontSize);
    end
end
%% PART 2


%% PART 3: Compare with sunspot number
% import sunspot data
sn_dir = fullfile(cfg.paths.sunspot_root, 'sn_ms_interp.dat');
sn_data = load(sn_dir);
sn_cr_lst = sn_data(:,1);
sn = sn_data(:,3);

% % plot figure: sunspot number series
% figure();
% LineWidth = 2;
% FontSize = 14;
% sz = 100;
% cross_0_lst = [1687,1823,1954,2128];
% cross_0_num = length(cross_0_lst);
% 
% plot(sn_cr_lst,sn,'k','LineWidth',LineWidth);
% hold on
% scatter(sn_cr_lst,sn,sz,sn_cr_lst,'filled');
% colormap(colormap_sc);
% hold on
% for i_cross = 1 : cross_0_num
%     xline(cross_0_lst(i_cross),'k','LineWidth',LineWidth);
% end
% ylabel('Sunspot Number');
% set(gca,'LineWidth',LineWidth,'FontSize',FontSize);
% subplot_xtick(-15,FontSize)

% % plot figure: Lissajours figure
% figure();
% LineWidth = 2;
% FontSize = 12;
% sz = 30;
% 
% for i_m = 1 : m_num
%     subplot(ceil(sqrt(m_num)),ceil(sqrt(m_num)),i_m);
%     scatter(sn,hc_sub(:,i_m),sz,sn_cr_lst,'filled');
%     grid on
% %     axis square
%     colormap(colormap_sc);
%     colorbar off;
% %     hold on
% %     plot(hc_sub(:,i_m),sn,'LineWidth',LineWidth);
% %     xlabel('Sunspot Number');
%     ylabel(['g_{',num2str(l),'}^{',num2str(m_sub(i_m)),'}']);
%     set(gca,'LineWidth',LineWidth,'FontSize',FontSize);
% end

% % plot figure: smoothed Lissajour figure
% figure();
% LineWidth = 2;
% FontSize = 14;
% sz = 30;
% smooth_win = 13/12*365.2422/27.2753; % from Carrington Rotation Period to Year
% 
% for i_m = 1 : 1%m_num
% %     subplot(ceil(sqrt(m_num)),ceil(sqrt(m_num)),i_m);
%     scatter(sn,smooth(hc_sub(:,i_m),smooth_win),sz,sn_cr_lst,'filled');
%     grid on
%     axis square
%     colormap(colormap_sc);
%     colorbar off;
% %     hold on
% %     plot(hc_sub(:,i_m),sn,'LineWidth',LineWidth);
%     xlabel('Sunspot Number');
%     ylabel(['g_{',num2str(l),'}^{',num2str(m_sub(i_m)),'}']);
%     set(gca,'LineWidth',LineWidth,'FontSize',FontSize);
% end
%% PART 3


%% functions
function M_norm = calmoment(M,l)
    Rs = 6.9634e8; % [m]
    mu0 = 4*pi*1e-7; % [T*m/A]
    M_norm = M*4*pi*Rs^(l+3)/mu0/(l+1);
end

function [wave,f] = wavelet(x)
    tlen = numel(x);
    fb = cwtfilterbank('SignalLength',tlen,'Wavelet','amor','SamplingFrequency',1,'FrequencyLimits',[0.001 1]);
    [wave,f] = wt(fb,x);
end

function subplot_hc(x,y,i_h,h_sub,h_space,LineWidth,FontSize,sz,colormap_sc,label)
    w_base = 0.15;
    w_sub = 0.8;
    h_base = 0.1;
    cross_0_lst = [1687,1823,1954,2128];
    cross_0_num = length(cross_0_lst);
%     sharp_lst = [1676,1679;1694,1697;1822,1825;1982,1985;2124,2127];
%     sharp_num = length(sharp_lst);

    subplot('Position',[w_base,h_base+(h_sub+h_space)*i_h,w_sub,h_sub])
    yline(0,'r','LineWidth',LineWidth,'Alpha',0.2)
    hold on
    for i_cross = 1 : cross_0_num
        xline(cross_0_lst(i_cross),'k','LineWidth',LineWidth);
    end
    hold on
    plot(x,y,'k','LineWidth',LineWidth);
    hold on; grid on
    scatter(x,y,sz,x,'filled');
    colormap(colormap_sc);
    hold on
%     for i_sharp = 1 : sharp_num
%        sharp_beg = sharp_lst(i_sharp,1);
%         sharp_end = sharp_lst(i_sharp,2);
%         x_shadow(sharp_beg,sharp_end);
%     end
    ylabel(label)
    set(gca,'LineWidth',LineWidth,'FontSize',FontSize,'XTickLabel',[],'XMinorTick','on')
    box on
end

function subplot_ps(x,y,c,i_h,h_sub,h_space,LineWidth,FontSize,label)
    w_base = 0.12;
    w_sub = 0.72;
    h_base = 0.1;
    cross_0_lst = [1687,1823,1954,2128];
    cross_0_num = length(cross_0_lst);

    c_size = size(c);
    c_smooth = zeros(c_size);
    for i_row = 1 : c_size(1)
        c_smooth(i_row,:) = smooth(c(i_row,:));
    end

    subplot('Position',[w_base,h_base+(h_sub+h_space)*i_h,w_sub,h_sub])
%     h = pcolor(x,y,abs(c));
    h = pcolor(x,y,abs(c_smooth));
    set(h,'LineStyle','none')
    shading interp
%     for i_cross = 1 : cross_0_num
%         xline(cross_0_lst(i_cross),'k','LineWidth',LineWidth);
%     end
    ylabel({label,'{\itT} (y)'})
    set(gca,'YScale','linear','LineWidth',LineWidth,'FontSize',FontSize,'TickDir','out','XTickLabel',[],'XMinorTick','on','YMinorTick','on');
    colormap jet; colorbar off
    hold on
    box on
end

function subplot_ps_mean(y,c,i_h,h_sub,h_space,LineWidth,FontSize)
    w_base = 0.84;
    w_sub = 0.12;
    h_base = 0.1;

    subplot('Position',[w_base,h_base+(h_sub+h_space)*i_h,w_sub,h_sub])
    c_mean = smooth(mean(abs(c),2));
    y_std = linspace(min(y),max(y),100);
    c_mean_interp = interp1(y,c_mean,y_std,'spline');
    c_max = max(c_mean_interp);
    y_max = y_std(c_mean_interp == c_max);
%     % style 1: plot graph
%     plot(c_mean_interp,y_std,'LineWidth',LineWidth);
%     grid on
%     text(c_max*1/2,y_max,['max@',num2str(roundn(y_max,-2))],'Color','r','FontSize',FontSize);
    % style 2: histogram graph
    barh(y_std,c_mean_interp,'FaceColor','#4DBEEE','EdgeColor','flat');
%     text(c_max*1/2,y_max,['max@',num2str(roundn(y_max,-2))],'Color','k','FontSize',FontSize);
    % figure properties
    ylim tight
    set(gca,'YScale','linear','LineWidth',LineWidth,'FontSize',FontSize,'XColor','none','YTickLabel',[],'YMinorTick','on','Box','off');
end

function subplot_xtick(y_text,FontSize)
%     set(gca,'XTick',[1600,1650,1700,1750,1800,1850,1900,1950,2000,2050,2100,2150,2200,2250], ...
%         'XTickLabel',{'CR','1650','1700','1750','1800','1850','1900','1950','2000','2050','2100','2150','2200','2250'}); % CR ticks
%     text([1600,1650,1700,1750,1800,1850,1900,1950,2000,2050,2100,2150,2200,2250], ...
%        zeros(1,14) + y_text, ...
%         {'yyyy/mm','1977/01','1980/10','1984/07','1988/03','1991/12','1995/09','1999/06','2003/03','2006/11','2010/08','2014/05','2018/02','2021/11'}, ...
%         'HorizontalAlignment','center','FontSize',FontSize); % date ticks
    set(gca,'XTick',[1600,1650,1700,1750,1800,1850,1900,1950,2000,2050,2100,2150,2200,2250], ...
        'XTickLabel',{'CR','1650','','1750','','1850','','1950','','2050','','2150','','2250'}); % CR ticks
    text([1580,1650,1750,1850,1950,2050,2150,2250], ...
       zeros(1,8) + y_text, ...
        {'yyyy/mm','1977/01','1984/07','1991/12','1999/06','2006/11','2014/05','2021/11'}, ...
        'HorizontalAlignment','center','FontSize',FontSize); % date ticks
end

function x_shadow(x_left,x_right)
    yl = ylim;
    y_down = yl(1); y_up = yl(2);
    vert = [x_left,y_down;x_right,y_down;x_right,y_up;x_left,y_up];
    f2 = [1,2,3,4];
    patch('Faces',f2,'Vertices',vert,'FaceColor','green','FaceAlpha',0.5,'EdgeColor','none');
end
