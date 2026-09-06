clear; close all;
cfg = project_config();
style = 2;
%% import data
omni_file = cfg.paths.omni_27day;
omni_data = load(omni_file);
year_omni = omni_data(:,1);
doy_omni = omni_data(:,2);
epoch_omni = YearDoy2Epoch(year_omni, doy_omni);
Pf = omni_data(:,29);

ssn_file = fullfile(cfg.paths.sunspot_root, 'SN_ms_tot_V2.0_202406.csv');
ssn_data = importdata(ssn_file);
ssn_data = ssn_data(2727:end-6,:); % recent 4 solar cycles
deci_year_ssn = ssn_data(:,3);
epoch_ssn = YearMonth2Epoch(deci_year_ssn);
ssn = ssn_data(:,4);
%% eliminate bad points
Pf(Pf>50) = nan;
%% interpolate to standard epoch (Sunspot Number)
epoch_std = epoch_ssn;
Pf_interp = interp1(epoch_omni, Pf, epoch_std);
Pf_interp = movmean(Pf_interp, 13); % 13-month smooth
%% mark solar maximum and solar minimum
max_ind = [46, 165, 309, 458];
min_ind = [127, 246, 394, 526];
%% plot figure
LineWidth = 2;
FontSize = 16;

if style == 1
    figure()
    plot(epoch_std, ssn, 'y', 'LineWidth', LineWidth)
    grid on
    datetick('x','yyyy')
    xlabel('YEAR')
    ylabel('Sunspot Number')
    set(gca, 'XColor', 'w', 'YColor', 'w','LineWidth', LineWidth, 'FontSize', FontSize)
    
    figure()
    plot(epoch_std, Pf_interp, 'c', 'LineWidth', LineWidth)
    grid on
    datetick('x','yyyy')
    xlabel('YEAR')
    ylabel('P_{dyn} [nPa]')
    set(gca, 'XColor', 'w', 'YColor', 'w', 'LineWidth', LineWidth, 'FontSize', FontSize)
elseif style == 2
    figure();
    yyaxis left
    plot(epoch_std, ssn, 'y', 'LineWidth', LineWidth)
    ylabel('Sunspot Number')
    set(gca, 'YColor', 'y')
    yyaxis right
    plot(epoch_std, Pf_interp, 'c', 'LineWidth', LineWidth)
    ylabel('P_{dyn} [nPa]')
    set(gca, 'YColor', 'c')
    grid on
    datetick('x','yyyy')
    xlabel('YEAR')
    set(gca, 'XColor', 'w', 'LineWidth', LineWidth, 'FontSize', FontSize)
elseif style == 3
    figure()
    subplot(2,1,1)
    plot(epoch_std, ssn, 'y', 'LineWidth', LineWidth)
    mark_max_min(epoch_std, max_ind, min_ind, 'y', LineWidth, FontSize)
    grid on
    datetick('x','yyyy')
    ylabel('Sunspot Number')
    set(gca, 'XColor', 'y', 'YColor', 'y','XTickLabel', [], 'LineWidth', LineWidth, 'FontSize', FontSize)
    
    subplot(2,1,2)
    plot(epoch_std, Pf_interp, 'c', 'LineWidth', LineWidth)
    mark_max_min(epoch_std, max_ind, min_ind, 'c', LineWidth, FontSize)
    grid on
    datetick('x','yyyy')
    xlabel('YEAR')
    ylabel('P_{dyn} [nPa]')
    set(gca, 'XColor', 'c', 'YColor', 'c', 'LineWidth', LineWidth, 'FontSize', FontSize)
end
%%



%% functions
function epoch = YearDoy2Epoch(year, doy)
% Converts year and day-of-year to epoch time
% Input:
%   year - year array
%   doy  - day-of-year array
% Output:
%   epoch - epoch array
    epoch = zeros(size(year));
    for i = 1:length(year)
        dateTime = datetime(year(i), 1, 1, 12, 0, 0) + caldays(doy(i) - 1);
        epoch(i) = datenum(dateTime);
    end
end

function epoch = YearMonth2Epoch(deci_year)
% Converts year and day-of-year to epoch time
% Input:
%   deci_year - a decimal array representing the year
% Output:
%   epoch - epoch array
    epoch = zeros(size(deci_year));
    for i = 1:length(deci_year)
        year = floor(deci_year(i));
        year_beg = datenum([year, 1, 1, 0, 0, 0]);
        year_end = datenum([year+1, 1, 1, 0, 0, 0]);
        frac_year = deci_year(i) - year;
        epoch(i) = year_beg + frac_year * (year_end - year_beg);
    end
end

function mark_max_min(epoch, max_ind, min_ind, color, LineWidth, FontSize)
    for i = 1 : length(max_ind)
        hold on
        xline(epoch(max_ind(i)), '--', 'Color', color, 'LineWidth', LineWidth, 'FontSize', FontSize);
        hold on
        xline(epoch(min_ind(i)), ':', 'Color', color, 'LineWidth', LineWidth, 'FontSize', FontSize)
        hold on
    end
end
