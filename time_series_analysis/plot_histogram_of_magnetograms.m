clear; close all;
cfg = project_config();
CR = 2097;
file_name = ['cr',num2str(CR)];
GONG_dir = [cfg.paths.gong_fits, filesep];
WSO_dir = [cfg.paths.wso_field, filesep];
GONG_recon_dir = [fullfile(cfg.output_root, 'harmonics_map', 'GONG'), filesep];
WSO_recon_dir = [fullfile(cfg.output_root, 'harmonics_map', 'WSO'), filesep];

GONG_map = importdata([GONG_dir,file_name,'.mat']); % [Gs]
WSO_map = importdata([WSO_dir,file_name,'.mat']); % [nT]
GONG_recon_map = importdata([GONG_recon_dir,file_name,'.mat']); % [Gs]
WSO_recon_map = importdata([WSO_recon_dir,file_name,'.mat']); % [nT]

figure();
histogram(GONG_map,'Normalization','probability','EdgeColor','none');
hold on
histogram(WSO_map./100,'Normalization','probability','FaceColor','none','EdgeColor','r','LineWidth',2);
legend('GONG map','WSO map')
xlim([-5 5])
set(gca,'LineWidth',2,'FontSize',20)
title(['CR',num2str(CR)])

figure();
histogram(GONG_recon_map,'Normalization','probability','EdgeColor','none');
hold on
histogram(WSO_recon_map./100,'Normalization','probability','FaceColor','none','EdgeColor','r','LineWidth',2);
legend('GONG reconstructed map','WSO reconstructed map')
xlim([-5 5])
set(gca,'LineWidth',2,'FontSize',20)
title(['CR',num2str(CR)])
