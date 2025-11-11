function dataset_path = resolve_dataset_path(dataset)
% Resolve dataset path: if relative, assume it's in python/ directory
% 
% Usage:
%   dataset_path = resolve_dataset_path('SJAFFE');
%   % Returns: '../python/SJAFFE'
%
%   dataset_path = resolve_dataset_path('/absolute/path/to/dataset');
%   % Returns: '/absolute/path/to/dataset' (unchanged)

    % Check if path is absolute
    is_absolute = false;
    
    % Windows absolute path (starts with drive letter)
    if ispc && length(dataset) >= 2 && dataset(2) == ':'
        is_absolute = true;
    % Unix/Mac absolute path (starts with /)
    elseif ~ispc && length(dataset) >= 1 && dataset(1) == filesep
        is_absolute = true;
    % Contains path separator and looks like absolute path
    elseif ~isempty(strfind(dataset, filesep)) && (ispc && ~isempty(strfind(dataset, ':')) || ~ispc)
        % Check if it starts with / or has : (Windows drive)
        if dataset(1) == filesep || (ispc && ~isempty(strfind(dataset, ':')))
            is_absolute = true;
        end
    end
    
    % If not absolute, prepend '../python/'
    if ~is_absolute
        dataset_path = fullfile('..', 'python', dataset);
    else
        dataset_path = dataset;
    end
end

