function varargout = util(action, varargin)
% Utility wrapper. Call as:
%   save_dict(dataset, scores, name)
%   out = load_dict(dataset, name)
%   out = load_npy(path)
% Direct helpers are also exposed as individual functions below.

switch action
    case 'save_dict'
        save_dict(varargin{:});
    case 'load_dict'
        varargout{1} = load_dict(varargin{:});
    case 'load_npy'
        varargout{1} = load_npy(varargin{:});
    otherwise
        error('Unknown action %s', action);
end

end

function save_dict(dataset, scores, name)
% Save scores dict to pickle using Python to maintain parity
    if ~endsWith(name, '.pkl')
        name = [name '.pkl'];
    end
    fpath = fullfile(dataset, name);
    try
        f = py.open(fpath, 'wb');
        % Convert MATLAB containers.Map or struct to Python dict
        pyd = matlab_to_pydict(scores);
        py.pickle.dump(pyd, f);
        f.close();
    catch ME
        error('save_dict failed: %s. Ensure MATLAB has Python configured.', ME.message);
    end
end

function out = load_dict(dataset, name)
% Load a pickle using Python and convert to MATLAB types
    if ~endsWith(name, '.pkl')
        if ~isfile(fullfile(dataset, name))
            name = [name '.pkl'];
        end
    end
    fpath = fullfile(dataset, name);
    try
        f = py.open(fpath, 'rb');
        obj = py.pickle.load(f);
        f.close();
        % Special handling: dict with integer keys -> cell array ordered by key
        if isa(obj, 'py.dict')
            try
                keys = cell(py.list(obj.keys()));
                ks = zeros(1, numel(keys));
                ok = true;
                for i=1:numel(keys)
                    try
                        ks(i) = double(keys{i});
                    catch
                        ok = false; break;
                    end
                end
                if ok
                    N = max(ks) + 1; % keys are 0-based
                    vals = cell(1, N);
                    for i=1:numel(keys)
                        k = ks(i);
                        vals{k+1} = py2mat(obj{keys{i}});
                    end
                    out = vals;
                else
                    out = py2mat(obj);
                end
            catch
                out = py2mat(obj);
            end
        else
            out = py2mat(obj);
        end
    catch ME
        error('load_dict failed: %s. Ensure MATLAB has Python configured.', ME.message);
    end
end

function A = load_npy(path)
% Load .npy via Python numpy and convert to MATLAB array
    try
        np = py.importlib.import_module('numpy');
        arr = np.load(path);
        % Ensure float64 and Fortran order for MATLAB reshape
        arr = np.array(arr, pyargs('dtype', np.float64, 'order', 'F'));
        shape = cell(py.tuple(arr.shape));
        if numel(shape) == 1
            m = double(shape{1});
            data = double(py.array.array('d', py.numpy.nditer(arr, pyargs('order','F'))));
            A = reshape(data, [m, 1]);
        elseif numel(shape) == 2
            m = double(shape{1}); n = double(shape{2});
            data = double(py.array.array('d', py.numpy.nditer(arr, pyargs('order','F'))));
            A = reshape(data, [m, n]);
        else
            % Fallback generic conversion
            A = py2mat(arr);
        end
    catch ME
        error('load_npy failed: %s. Ensure numpy is installed for MATLAB Python.', ME.message);
    end
end

function pyd = matlab_to_pydict(m)
    if isa(m, 'containers.Map')
        keys = m.keys; vals = m.values;
        pyd = py.dict();
        for i = 1:numel(keys)
            k = keys{i}; v = vals{i};
            pyd{mat2py(k)} = mat2py(v);
        end
    elseif isstruct(m)
        fn = fieldnames(m);
        pyd = py.dict();
        for i = 1:numel(fn)
            k = fn{i}; v = m.(k);
            pyd{k} = mat2py(v);
        end
    else
        error('Unsupported type for matlab_to_pydict');
    end
end

function y = mat2py(x)
% Convert MATLAB basic types to Python objects
    if isnumeric(x) || islogical(x)
        y = py.numpy.array(x);
    elseif ischar(x) || isstring(x)
        y = char(x);
    elseif iscell(x)
        pylist = py.list();
        for i = 1:numel(x)
            pylist.append(mat2py(x{i}));
        end
        y = pylist;
    elseif isa(x, 'containers.Map') || isstruct(x)
        y = matlab_to_pydict(x);
    else
        error('Unsupported type in mat2py');
    end
end

function m = py2mat(x)
% Recursively convert Python types to MATLAB types
    if isa(x, 'py.numpy.ndarray')
        if x.ndim == int32(0)
            m = double(x.tolist());
        else
            m = cell2mat_recursive(x.tolist());
        end
    elseif isa(x, 'py.list') || isa(x, 'py.tuple')
        n = int64(py.len(x));
        c = cell(n,1);
        for i=1:n
            c{i} = py2mat(x{int32(i-1)});
        end
        % Try to concatenate if uniform
        try
            m = cell2mat(c);
        catch
            m = c;
        end
    elseif isa(x, 'py.dict')
        keys = cell(py.list(x.keys()));
        m = struct();
        for i=1:numel(keys)
            k = string(keys{i});
            m.(matlab.lang.makeValidName(k)) = py2mat(x{keys{i}});
        end
    elseif isa(x, 'py.float') || isa(x, 'py.int') || isa(x, 'py.long')
        m = double(x);
    elseif ischar(x) || isa(x,'py.str')
        m = string(char(x));
    else
        % Fallback: try tolist
        try
            m = cell2mat_recursive(x.tolist());
        catch
            % As-is
            m = x;
        end
    end
end

function M = cell2mat_recursive(c)
% Convert nested Python lists to MATLAB numeric array when possible
    if iscell(c)
        try
            M = cellfun(@cell2mat_recursive, c, 'UniformOutput', false);
            M = cell2mat(M);
        catch
            % Not rectangular
            M = c;
        end
    else
        M = double(c);
    end
end
