%SETUP_MATLAB Add this repository and its subdirectories to the MATLAB path.

repo_root = fileparts(mfilename('fullpath'));
addpath(genpath(repo_root));
disp('Added Synoptic Map Prediction repository to the MATLAB path:');
disp(repo_root);
