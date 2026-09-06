clear; close all;
cfg = project_config();
save_or_not = 0;
%% PART 1: import data
store_dir = [cfg.paths.sunspot_root, filesep];
spot_name = 'SN_ms_tot_V2.0.csv';
data_full = importdata([store_dir,spot_name]);
frac_lst = data_full(:,3);
sn = data_full(:,4);
sn_smooth = smooth(sn,12);
%% plot figure
figure();
LineWidth = 2;
FontSize = 15;
plot(frac_lst,sn,'LineWidth',LineWidth);
hold on
plot(frac_lst,sn_smooth,'LineWidth',LineWidth);
daspect([1 4 1])
xlabel('Year');
ylabel('Sunspot Number');
legend('Monthly Mean','Smoothed','Location','best');
set(gca,'LineWidth',LineWidth,'FontSize',FontSize);
%% PART 1


%% PART 2: construct Carrington Rotation series
CR_beg = 1642;
CR_end = 2258;
CR_lst = CR_beg : CR_end;
CR_num = length(CR_lst);
CR_frac_lst = zeros(CR_num,1);
% transform to fraction of year
CR_beg_date = datetime(1976,05,27);
CR_beg_doy = day(CR_beg_date,'dayofyear');
year2day = 365.2422; % one year
CR2day = 27.2753; % one Carrington Rotation
CR_beg_frac = 1976 + CR_beg_doy ./ year2day; % beginning in fraction of year
for i_cr = 1 : CR_num
    CR_frac_lst(i_cr) = CR_beg_frac + (i_cr-1).*CR2day./year2day;
end
% interpolate into CR series
sn_interp = interp1(frac_lst,sn,CR_frac_lst,'linear');
%% plot figure
data = data_full(2725:end,:); % Year 1976 at index 2725
figure();
LineWidth = 2;
FontSize = 15;
plot(CR_frac_lst,sn_interp,'LineWidth',LineWidth);
xlabel('Year');
ylabel('Sunspot Number');
set(gca,'LineWidth',LineWidth,'FontSize',FontSize);
%% save data
save_dir = [cfg.paths.sunspot_root, filesep];
save_var = [CR_lst.',CR_frac_lst,sn_interp];
if save_or_not == 1
    save_file = [save_dir,'sn_ms_interp.dat'];
    save(save_file,'save_var','-ascii');
    disp('Neaten Sunspot Number Successfully');
    
    save_file = [save_dir,'sn_ms_interp.csv'];
    csvwrite(save_file,sn_interp);
end
%% PART 2
