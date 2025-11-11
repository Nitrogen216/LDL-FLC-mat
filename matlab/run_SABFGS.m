function run_SABFGS_all(datasets)
% MATLAB port of python/run_SABFGS.py
% Runs SA-BFGS (bfgs_ldl) algorithm with 10-fold cross-validation
%
% Usage:
%   run_SABFGS_all()                  % Run on default datasets
%   run_SABFGS_all(datasets)          % Run on specified datasets
%
% Example:
%   run_SABFGS_all({'SJAFFE'})
%   run_SABFGS_all({'SJAFFE', 'M2B', 'RAF_ML'})
    init_path(); % Initialize paths to core modules
    
    if nargin < 1 || isempty(datasets)
        datasets = {"SJAFFE", "M2B", "Movie", "RAF_ML", "Flickr_ldl", "Ren", "fbp5500", ...
                    "Gene", "SBU_3DFE", "SCUT_FBP", "Scene", "Twitter_ldl"};
    end
    
    for i = 1:numel(datasets)
        run_KF(datasets{i});
    end
end

function run_KF(dataset)
    fprintf('%s\n', dataset);
    
    % Handle dataset path: if not absolute, assume it's in python/ directory
    dataset = resolve_dataset_path(dataset);
    
    % Load data
    X = util('load_npy', fullfile(dataset, 'feature.npy'));
    Y = util('load_npy', fullfile(dataset, 'label.npy'));
    train_inds = util('load_dict', dataset, 'train_inds');
    test_inds = util('load_dict', dataset, 'test_inds');
    
    scores = containers.Map();
    
    % 10-fold cross-validation
    for i = 1:10
        fprintf('%s fold %d\n', dataset, i);
        
        % Get train/test split (convert from 0-based to 1-based indexing)
        tr = train_inds{i}; te = test_inds{i};
        train_x = X(tr+1, :); train_y = Y(tr+1, :);
        test_x = X(te+1, :);  test_y = Y(te+1, :);
        
        % Run SA-BFGS
        scores = run_sabfgs_fold(i, dataset, train_x, train_y, test_x, test_y, scores);
    end
    
    % Save results
    util('save_dict', dataset, scores, 'BFGSLDL1.pkl');
    fprintf('Results saved to %s\n', fullfile(dataset, 'BFGSLDL1.pkl'));
end

function scores = run_sabfgs_fold(i, dataset, train_x, train_y, test_x, test_y, scores)
    % Run bfgs_ldl model on a single fold
    
    % Create and train model (C=0 by default)
    model = bfgs_ldl(0);
    model.fit(train_x, train_y);
    
    % Predict
    y_pre = model.predict(test_x);
    
    % Evaluate
    [cheby, clark, can, kl, cosine, inter] = ldl_metrics('score', test_y, y_pre);
    val = [cheby, clark, can, kl, cosine, inter];
    
    % Store results
    key = char(model);
    if ~isKey(scores, key)
        scores(key) = [];
    end
    scores(key) = [scores(key); val];
    
    % Print results for this fold
    fprintf('  Fold %d: Cheby=%.4f, Clark=%.4f, Canberra=%.4f, KL=%.4f, Cosine=%.4f, Inter=%.4f\n', ...
            i, cheby, clark, can, kl, cosine, inter);
end

%% Optional: Run with different regularization parameters
function run_SABFGS_with_params(dataset, C_values)
% Run SA-BFGS with different regularization parameters
%
% Usage:
%   run_SABFGS_with_params('SJAFFE', [0, 0.001, 0.01, 0.1, 1])

    if nargin < 2
        C_values = [0, 0.001, 0.01, 0.1, 1, 10];
    end
    
    fprintf('%s - testing %d C values\n', dataset, numel(C_values));
    
    % Load data
    X = util('load_npy', fullfile(dataset, 'feature.npy'));
    Y = util('load_npy', fullfile(dataset, 'label.npy'));
    train_inds = util('load_dict', dataset, 'train_inds');
    test_inds = util('load_dict', dataset, 'test_inds');
    
    all_scores = containers.Map();
    
    % Test each C value
    for c_idx = 1:numel(C_values)
        C = C_values(c_idx);
        fprintf('\nTesting C = %.4g\n', C);
        
        scores = containers.Map();
        
        % 10-fold cross-validation
        for i = 1:10
            fprintf('  Fold %d\n', i);
            
            tr = train_inds{i}; te = test_inds{i};
            train_x = X(tr+1, :); train_y = Y(tr+1, :);
            test_x = X(te+1, :);  test_y = Y(te+1, :);
            
            % Train model with specific C
            model = bfgs_ldl(C);
            model.fit(train_x, train_y);
            y_pre = model.predict(test_x);
            
            % Evaluate
            [cheby, clark, can, kl, cosine, inter] = ldl_metrics('score', test_y, y_pre);
            val = [cheby, clark, can, kl, cosine, inter];
            
            key = char(model);
            if ~isKey(scores, key)
                scores(key) = [];
            end
            scores(key) = [scores(key); val];
        end
        
        % Store in master results
        key = sprintf('bfgs_ldl_%.4g', C);
        all_scores(key) = scores;
        
        % Print average results for this C value
        if ~isempty(scores.keys)
            k = scores.keys; k = k{1};
            vals = scores(k);
            avg_vals = mean(vals, 1);
            fprintf('  Average: Cheby=%.4f, Clark=%.4f, KL=%.4f\n', ...
                    avg_vals(1), avg_vals(2), avg_vals(4));
        end
    end
    
    % Save all results
    util('save_dict', dataset, all_scores, 'BFGSLDL_param_search.pkl');
    fprintf('\nAll results saved to %s\n', fullfile(dataset, 'BFGSLDL_param_search.pkl'));
end

