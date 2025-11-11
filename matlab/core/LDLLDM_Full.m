classdef LDLLDM_Full < handle
    % MATLAB port of LDLLDM.py
    % Label Distribution Learning with Label Distribution Manifold
    properties
        X          % augmented feature matrix with intercept
        Y          % label distribution matrix
        l1         % L2 regularization
        l2         % global manifold weight
        l3         % local manifold weight
        g          % number of clusters
        n_examples
        n_features
        n_outputs
        clusters   % cell array of Cluster objects
        inds       % cell array of logical indices
        W          % learned weight matrix
    end
    
    methods
        function obj = LDLLDM_Full(X, Y, l1, l2, l3, g, clu_labels, manifolds)
            if nargin < 6, g = 0; end
            if nargin < 7, clu_labels = []; end
            if nargin < 8, manifolds = []; end
            
            obj.X = append_intercept(X);
            obj.Y = Y;
            obj.l1 = l1;
            obj.l2 = l2;
            obj.l3 = l3;
            obj.g = g;
            
            [obj.n_examples, obj.n_features] = size(obj.X);
            obj.n_outputs = size(obj.Y, 2);
            
            % K-means clustering if labels not provided
            if isempty(clu_labels) && g > 0
                clu_labels = kmeans(Y, g) - 1; % 0-based for consistency
            end
            
            obj.init_clusters(clu_labels, manifolds);
        end
        
        function init_clusters(obj, clu_labels, manifolds)
            obj.clusters = {};
            obj.inds = {};
            
            % Global label distribution manifold
            clu = LDLLDM_Cluster(obj.X, obj.l2, obj.Y, []);
            obj.clusters{end+1} = clu;
            obj.inds{end+1} = true(obj.n_examples, 1);
            
            % Local manifolds per cluster
            if obj.g > 1
                for i = 0:(obj.g-1)
                    ind = (clu_labels == i);
                    X_i = obj.X(ind, :);
                    Y_i = obj.Y(ind, :);
                    
                    if isempty(manifolds)
                        clu = LDLLDM_Cluster(X_i, obj.l3, Y_i, []);
                    else
                        clu = LDLLDM_Cluster(X_i, obj.l3, Y_i, manifolds{i+1});
                    end
                    
                    obj.clusters{end+1} = clu;
                    obj.inds{end+1} = ind;
                end
            end
        end
        
        function [loss, gradvec] = LDL(obj, Wvec)
            W = reshape(Wvec, obj.n_features, obj.n_outputs);
            Y_hat = softmax(obj.X * W);
            
            [loss, grad] = KL(obj.Y, Y_hat, []);
            grad = obj.X' * grad;
            
            if obj.l1 ~= 0
                loss = loss + 0.5 * obj.l1 * sum(W(:).^2);
                grad = grad + obj.l1 * W;
            end
            
            % Add manifold regularization from all clusters
            for k = 1:numel(obj.clusters)
                ind = obj.inds{k};
                clu = obj.clusters{k};
                [clu_l, clu_g] = clu.LDL(Y_hat(ind, :));
                loss = loss + clu_l;
                grad = grad + clu_g;
            end
            
            gradvec = grad(:);
        end
        
        function solve(obj, max_iters)
            if nargin < 2, max_iters = 600; end
            
            W0 = eye(obj.n_features, obj.n_outputs);
            W0 = W0(:);
            
            opts = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
                'SpecifyObjectiveGradient', true, 'Display', 'off', ...
                'MaxIterations', max_iters);
            
            fun = @(w) obj.LDL(w);
            Wopt = fminunc(fun, W0, opts);
            obj.W = reshape(Wopt, obj.n_features, obj.n_outputs);
        end
        
        function Yhat = predict(obj, X_test)
            X_test_aug = append_intercept(X_test);
            Yhat = softmax(X_test_aug * obj.W);
        end
        
        function s = char(obj)
            s = sprintf('LDLLDM_%.4g_%.4g_%.4g', obj.l1, obj.l2, obj.l3);
        end
    end
end


%% Helper functions
function Y = softmax(Z)
    Z = Z - max(Z, [], 2);
    E = exp(Z);
    S = sum(E, 2);
    Y = E ./ S;
end

function [loss, grad] = KL(y, y_hat, J)
    epsv = 1e-15;
    y = max(min(y, 1), epsv);
    y_hat = max(min(y_hat, 1), epsv);
    
    if isempty(J)
        loss = -sum(y .* log(y_hat), 'all');
        grad = y_hat - y;
    else
        loss = -sum(J .* y .* log(y_hat), 'all');
        grad = J .* (y_hat - y);
    end
end

function X_aug = append_intercept(X)
    X_aug = [X, ones(size(X, 1), 1)];
end

