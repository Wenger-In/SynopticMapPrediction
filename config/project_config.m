function cfg = project_config()
%PROJECT_CONFIG Load repository paths shared by MATLAB scripts.

config_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(config_dir);
raw = jsondecode(fileread(fullfile(config_dir, 'default.json')));

local_file = fullfile(config_dir, 'local.json');
if isfile(local_file)
    local = jsondecode(fileread(local_file));
    raw = merge_struct(raw, local);
end

data_root = getenv('SMP_DATA_ROOT');
if isempty(data_root)
    data_root = resolve_path(repo_root, raw.data_root);
end
output_root = getenv('SMP_OUTPUT_ROOT');
if isempty(output_root)
    output_root = resolve_path(repo_root, raw.output_root);
end

cfg = raw;
cfg.repo_root = repo_root;
cfg.data_root = data_root;
cfg.output_root = output_root;
names = fieldnames(raw.paths);
for i = 1:numel(names)
    value = raw.paths.(names{i});
    if startsWith(value, 'data/')
        cfg.paths.(names{i}) = fullfile(data_root, extractAfter(value, 'data/'));
    elseif startsWith(value, 'outputs/')
        cfg.paths.(names{i}) = fullfile(output_root, extractAfter(value, 'outputs/'));
    else
        cfg.paths.(names{i}) = resolve_path(repo_root, value);
    end
end
end
function value = resolve_path(root, value)
if ispc && ~isempty(regexp(value, '^[A-Za-z]:', 'once'))
    return
end
if startsWith(value, filesep)
    return
end
value = fullfile(root, value);
end

function merged = merge_struct(base, override)
merged = base;
names = fieldnames(override);
for i = 1:numel(names)
    name = names{i};
    if isstruct(override.(name)) && isfield(base, name) && isstruct(base.(name))
        merged.(name) = merge_struct(base.(name), override.(name));
    else
        merged.(name) = override.(name);
    end
end
end
