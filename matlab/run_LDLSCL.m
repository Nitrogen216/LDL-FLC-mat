function run_LDLSCL_all(datasets, mode)
% MATLAB port of python/run_LDLSCL.py
% Runs LDL-SCL algorithm with 10-fold cross-validation
% 
% Usage:
%   run_LDLSCL_all()                    % Run on default datasets
%   run_LDLSCL_all(datasets)            % Run on specified datasets
%   run_LDLSCL_all(datasets, 'tune')    % Run with parameter tuning
%
% Example:
%   run_LDLSCL_all({'SJAFFE'}, 'run')
%   run_LDLSCL_all({'SJAFFE'}, 'tune')
    init_path(); % Initialize paths to core modules
    
    if nargin < 1 || isempty(datasets)
        datasets = {"SJAFFE", "M2B", "Movie", "RAF_ML", "Flickr_ldl", "Ren", "fbp5500", ...
                    "Gene", "SBU_3DFE", "SCUT_FBP", "Scene", "Twitter_ldl"};
    end
    
    if nargin < 2
        mode = 'run'; % 'run' or 'tune'
    end
    
    for i = 1:numel(datasets)
        run_KF(datasets{i}, mode);
    end
end

function run_KF(dataset, mode)
    fprintf('%s\n', dataset);
    
    % Handle dataset path: if not absolute, assume it's in python/ directory
    dataset = resolve_dataset_path(dataset);
    
    X = util('load_npy', fullfile(dataset, 'feature.npy'));
    Y = util('load_npy', fullfile(dataset, 'label.npy'));
    train_inds = util('load_dict', dataset, 'train_inds');
    test_inds = util('load_dict', dataset, 'test_inds');
    
    scores = containers.Map();
    
    for i = 1:10
        fprintf('%s fold %d\n', dataset, i);
        tr = train_inds{i}; te = test_inds{i};
        train_x = X(tr+1, :); train_y = Y(tr+1, :);
        test_x = X(te+1, :);  test_y = Y(te+1, :);
        
        if strcmp(mode, 'tune')
            scores = tune_LDL_SCL(i, dataset, train_x, train_y, test_x, test_y, scores);
        else
            scores = run_LDLSCL_fold(i, dataset, train_x, train_y, test_x, test_y, scores);
        end
    end
    
    % Save results
    util('save_dict', dataset, scores, 'LDLSCL.pkl');
    fprintf('Results saved to %s\n', fullfile(dataset, 'LDLSCL.pkl'));
end

function scores = run_LDLSCL_fold(i, dataset, train_x, train_y, test_x, test_y, scores)
    % Run with pre-configured parameters for each dataset
    
    if ismember(dataset, ["Gene", "Movie"])
        l1 = 1e-5;
        l2 = 1e-3;
        l3 = 1e-4;
        c = 12;
    elseif ismember(dataset, ["M2B", "SCUT_FBP", "SBU_3DFE", "Scene", "SJAFFE"])
        l1 = 1e-4;
        l2 = 1e-3;
        l3 = 1e-3;
        if strcmp(dataset, "SJAFFE")
            c = 5;
        else
            c = 8;
        end
    elseif ismember(dataset, ["fbp5500", "RAF_ML", "Twitter_ldl", "Flickr_ldl", "Ren"])
        l1 = 1e-3;
        l2 = 1e-3;
        l3 = 1e-3;
        c = 5;
    else
        % Default parameters
        l1 = 1e-3;
        l2 = 1e-3;
        l3 = 1e-3;
        c = 5;
    end
    
    % Run LDL_SCL
    y_pre = LDL_SCL(train_x, train_y, test_x, test_y, l1, l2, l3, c);
    key = sprintf('LDLSCL_%.5g_%.5g_%.5g_%d', l1, l2, l3, c);
    
    [cheby, clark, can, kl, cosine, inter] = ldl_metrics('score', test_y, y_pre);
    val = [cheby, clark, can, kl, cosine, inter];
    
    if ~isKey(scores, key)
        scores(key) = [];
    end
    scores(key) = [scores(key); val];
    
    fprintf('  %s: Cheby=%.4f, Clark=%.4f, KL=%.4f\n', key, cheby, clark, kl);
end

function scores = tune_LDL_SCL(i, dataset, train_x, train_y, test_x, test_y, scores)
    % Parameter tuning mode - grid search over hyperparameters
    % Note: This can be very time-consuming
    
    L1 = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
    L2 = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
    L3 = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
    groups = 0:13; % 14 groups
    
    % Generate all parameter combinations
    total_configs = numel(L1) * numel(L2) * numel(L3) * numel(groups);
    fprintf('Total configurations to test: %d\n', total_configs);
    
    finished = 0;
    
    % Grid search (can be slow - consider using parfor for parallelization)
    for l1 = L1
        for l2 = L2
            for l3 = L3
                for c = groups
                    finished = finished + 1;
                    
                    % Progress report every 500 iterations
                    if mod(finished, 500) == 0
                        fprintf('%s fold %d: %d/%d\n', dataset, i, finished, total_configs);
                    end
                    
                    try
                        % Run LDL_SCL with current parameters
                        y_pre = LDL_SCL(train_x, train_y, test_x, test_y, l1, l2, l3, c);
                        key = sprintf('LDLSCL_%.5g_%.5g_%.5g_%d', l1, l2, l3, c);
                        
                        [cheby, clark, can, kl, cosine, inter] = ldl_metrics('score', test_y, y_pre);
                        val = [cheby, clark, can, kl, cosine, inter];
                        
                        if ~isKey(scores, key)
                            scores(key) = [];
                        end
                        scores(key) = [scores(key); val];
                    catch ME
                        fprintf('  Error with params l1=%.5g l2=%.5g l3=%.5g c=%d: %s\n', ...
                                l1, l2, l3, c, ME.message);
                    end
                end
            end
        end
    end
    
    fprintf('Tuning completed: %d/%d configurations tested\n', finished, total_configs);
end

%% Parallelized version (optional - requires Parallel Computing Toolbox)
% Uncomment to use parallel processing for tuning
%
% function scores = tune_LDL_SCL_parallel(i, dataset, train_x, train_y, test_x, test_y, scores)
%     L1 = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
%     L2 = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
%     L3 = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
%     groups = 0:13;
%     
%     % Generate all parameter combinations
%     [L1g, L2g, L3g, Gg] = ndgrid(L1, L2, L3, groups);
%     params = [L1g(:), L2g(:), L3g(:), Gg(:)];
%     
%     results = cell(size(params, 1), 1);
%     
%     parfor idx = 1:size(params, 1)
%         l1 = params(idx, 1);
%         l2 = params(idx, 2);
%         l3 = params(idx, 3);
%         c = params(idx, 4);
%         
%         try
%             y_pre = LDL_SCL(train_x, train_y, test_x, test_y, l1, l2, l3, c);
%             key = sprintf('LDLSCL_%.5g_%.5g_%.5g_%d', l1, l2, l3, c);
%             [cheby, clark, can, kl, cosine, inter] = ldl_metrics('score', test_y, y_pre);
%             results{idx} = struct('key', key, 'val', [cheby, clark, can, kl, cosine, inter]);
%         catch
%             results{idx} = [];
%         end
%     end
%     
%     % Aggregate results
%     for idx = 1:numel(results)
%         if ~isempty(results{idx})
%             key = results{idx}.key;
%             val = results{idx}.val;
%             if ~isKey(scores, key)
%                 scores(key) = [];
%             end
%             scores(key) = [scores(key); val];
%         end
%     end
% end

