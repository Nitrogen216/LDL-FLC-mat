function T = save_ldlfc_metrics(dataset, out_csv)
% Run LDL-FC on a dataset and save per-fold metrics to CSV
% Columns: cheby, clark, can, kl, cosine, inter
if nargin<1, dataset = 'python/SJAFFE'; end
if nargin<2, out_csv = fullfile(dataset, 'ldlfc_matlab_metrics.csv'); end
init_path(); % Initialize paths to core modules

X = util('load_npy', fullfile(dataset, 'feature.npy'));
Y = util('load_npy', fullfile(dataset, 'label.npy'));
train_inds = util('load_dict', dataset, 'train_inds');
test_inds = util('load_dict', dataset, 'test_inds');

M = zeros(10,6);
for i=1:10
    tr = train_inds{i}; te = test_inds{i};
    train_x = X(tr+1, :); train_y = Y(tr+1, :);
    test_x = X(te+1, :);  test_y = Y(te+1, :);
    mdl = LDL_FLC(5, 1e-3, 1e-2);
    mdl.fit(train_x, train_y);
    mdl.solve();
    Yhat = mdl.predict(test_x);
    [cheby, clark, can, kl, cosine, inter] = ldl_metrics('score', test_y, Yhat);
    M(i,:) = [cheby, clark, can, kl, cosine, inter];
end
T = array2table(M, 'VariableNames', {'cheby','clark','can','kl','cosine','inter'});
writetable(T, out_csv);
fprintf('Saved MATLAB metrics to %s\n', out_csv);
end
