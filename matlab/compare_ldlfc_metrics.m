function D = compare_ldlfc_metrics(dataset, py_csv, m_csv, tol)
% Compare MATLAB and Python LDL-FC metrics CSVs; print max abs diff per column
init_path(); % Initialize paths to core modules
if nargin<1, dataset = 'python/SJAFFE'; end
if nargin<2, py_csv = fullfile(dataset, 'ldlfc_python_metrics.csv'); end
if nargin<3, m_csv = fullfile(dataset, 'ldlfc_matlab_metrics.csv'); end
if nargin<4, tol = 1e-6; end

TP = readtable(py_csv);
TM = readtable(m_csv);
if height(TP) ~= height(TM)
    error('Row count mismatch: Python %d vs MATLAB %d', height(TP), height(TM));
end
vars = TP.Properties.VariableNames;
D = table();
for k = 1:numel(vars)
    v = vars{k};
    dp = TP.(v) - TM.(v);
    D.(v) = abs(dp);
end
maxdiff = varfun(@max, D);
disp('Max abs diff per metric:');
disp(maxdiff);
ok = all(table2array(maxdiff) < tol, 'all');
if ok
    fprintf('Metrics match within tol=%g\n', tol);
else
    fprintf('Metrics differ beyond tol=%g\n', tol);
end
end

