function init_path()
% Initialize MATLAB paths for LDL-FLC project
% Call this function at the beginning of any script to ensure all modules are accessible
%
% Usage:
%   init_path();
%   % Now you can use LDL_FLC, LDL_LRR, etc.

    % Get the directory where this script is located
    matlab_dir = fileparts(mfilename('fullpath'));
    
    % Add matlab directory to path (if not already there)
    if ~contains(path, matlab_dir)
        addpath(matlab_dir);
    end
    
    % Add core directory to path (if not already there)
    core_dir = fullfile(matlab_dir, 'core');
    if ~contains(path, core_dir)
        addpath(core_dir);
    end
    
    fprintf('LDL-FLC paths initialized:\n');
    fprintf('  MATLAB dir: %s\n', matlab_dir);
    fprintf('  Core dir: %s\n', core_dir);
end

