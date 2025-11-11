function run_LDLLRR_all()
    init_path(); % Initialize paths to core modules
    datasets = {"SJAFFE", "M2B", "Movie", "RAF_ML", "Flickr_ldl", "Ren", "fbp5500", ...
                "Gene", "SBU_3DFE", "SCUT_FBP", "Scene", "Twitter_ldl"};
    for i=1:numel(datasets)
        run_KF(datasets{i});
    end
end

function run_KF(dataset)
    % Handle dataset path: if not absolute, assume it's in python/ directory
    dataset = resolve_dataset_path(dataset);
    
    X = util('load_npy', fullfile(dataset, 'feature.npy'));
    Y = util('load_npy', fullfile(dataset, 'label.npy'));
    train_inds = util('load_dict', dataset, 'train_inds');
    test_inds = util('load_dict', dataset, 'test_inds');
    scores = containers.Map();
    for i=1:10
        fprintf('%s fold %d\n', dataset, i);
        tr = train_inds{i}; te = test_inds{i};
        train_x = X(tr+1,:); train_y = Y(tr+1,:);
        test_x = X(te+1,:);  test_y = Y(te+1,:);
        scores = run_LDLLRR_fold(i, dataset, train_x, train_y, test_x, test_y, scores);
    end
    % Save as pickle for parity
    util('save_dict', dataset, scores, 'LDLLRR.pkl');
end

function scores = run_LDLLRR_fold(~, ~, train_x, train_y, test_x, test_y, scores)
    Lam = [1e-3]; Beta = [1];
    for lam = Lam
        for beta = Beta
            key = sprintf('LDLLRR_%.4g_%.4g', lam, beta);
            model = LDL_LRR('lam', lam, 'beta', beta);
            model.fit(train_x, train_y);
            y_pre = model.predict(test_x);
            [cheby, clark, can, kl, cosine, inter] = ldl_metrics('score', test_y, y_pre);
            if ~isKey(scores, key)
                scores(key) = [];
            end
            scores(key) = [scores(key); cheby, clark, can, kl, cosine, inter];
        end
    end
end

