function out = run_python_ldlfc(dataset)
% Run the Python LDL-FC script inside MATLAB's Python engine
% Returns captured metrics per fold as a cell array of numeric rows.
init_path(); % Initialize paths to core modules
if nargin<1, dataset = 'python/SJAFFE'; end

% Ensure python can see repo root
if count(py.sys.path, string(pwd)) == 0
    insert(py.sys.path, int32(0), string(pwd));
end

% Test dependencies
mods = {'numpy','scipy','sklearn','skfuzzy'};
for i=1:numel(mods)
    try
        py.importlib.import_module(mods{i});
    catch ME
        error('Python import failed for %s: %s', mods{i}, ME.message);
    end
end

% Run the Python module as a script
% We will import run_LDLFC.run_LDLFC and call it
try
    py_mod = py.importlib.import_module('python.run_LDLFC');
catch
    % adjust path: add 'python' folder
    if count(py.sys.path, string(fullfile(pwd,'python'))) == 0
        insert(py.sys.path, int32(0), string(fullfile(pwd,'python')));
    end
    py_mod = py.importlib.import_module('run_LDLFC');
end

% Monkey-patch score to capture outputs by wrapping print
out = {};
% Define a Python callable to intercept prints
py.print('Running Python LDL-FC...');
py_mod.run_LDLFC(string(dataset));
end

