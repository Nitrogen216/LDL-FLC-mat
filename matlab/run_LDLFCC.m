function run_LDLFCC(dataset)
% MATLAB equivalent of python/run_LDLFCC.py
    init_path(); % Initialize paths to core modules
    fprintf('%s\n', dataset);
    
    % Handle dataset path: if not absolute, assume it's in python/ directory
    dataset = resolve_dataset_path(dataset);
    
    X = util('load_npy', fullfile(dataset, 'feature.npy'));
    Y = util('load_npy', fullfile(dataset, 'label.npy'));
    train_inds = util('load_dict', dataset, 'train_inds');
    test_inds = util('load_dict', dataset, 'test_inds');

    for i = 1:10
        fprintf('training %d fold\n', i);
        tr = train_inds{i}; te = test_inds{i};
        train_x = X(tr+1, :); train_y = Y(tr+1, :);
        test_x = X(te+1, :);  test_y = Y(te+1, :);
        run_fold(i-1, train_x, train_y, test_x, test_y);
    end
end

function run_fold(~, train_x, train_y, test_x, test_y)
    l1 = 1e-3; l2 = 1e-2; g = 5;
    [U, manifolds] = joint_FCLC('get_fuzzy_manifolds', train_x, train_y, g);
    model = LDL_FLC(g, l1, l2);
    model.fit(train_x, train_y, U, manifolds);
    model.solve();
    Yhat = model.predict(test_x);
    [cheby, clark, can, kl, cosine, inter] = ldl_metrics('score', test_y, Yhat);
    disp([cheby, clark, can, kl, cosine, inter]);
end

