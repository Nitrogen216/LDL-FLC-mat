function Z = barycenter_kneighbors_graph(X, n_neighbors, reg)
% MATLAB equivalent of Python barycenter_kneighbors_graph
% X: n x d, returns Z: n x n row-stochastic weight matrix
    if nargin < 2 || isempty(n_neighbors)
        n_neighbors = size(X,1) - 1;
    end
    if nargin < 3 || isempty(reg)
        reg = 1e-3;
    end
    % Find neighbors (exclude self)
    if exist('knnsearch','file') == 2
        [idx, ~] = knnsearch(X, X, 'K', n_neighbors+1);
        idx = idx(:,2:end);
    else
        % Fallback: brute-force neighbors without toolboxes
        n = size(X,1);
        idx = zeros(n, n_neighbors);
        for i=1:n
            xi = X(i,:);
            % compute squared distances to all
            diffs = X - xi;
            d2 = sum(diffs.^2, 2);
            d2(i) = inf; % exclude self
            [~, order] = sort(d2, 'ascend');
            idx(i,:) = order(1:n_neighbors);
        end
    end
    % Compute barycenter weights per row
    B = barycenter_weights(X, idx, reg);
    n = size(X,1);
    Z = zeros(n,n);
    for i=1:n
        Z(i, idx(i,:)) = B(i,:);
    end
end

function B = barycenter_weights(X, ind, reg)
% Solve for weights w for each sample using local covariance regularization
    [n_samples, n_neighbors] = size(ind);
    d = size(X,2);
    B = zeros(n_samples, n_neighbors);
    v = ones(n_neighbors,1);
    for i=1:n_samples
        Zi = X(ind(i,:),:)'; % d x k
        Ci = Zi - X(i,:)';   % d x k
        G = Ci' * Ci;        % k x k
        traceG = trace(G);
        if traceG > 0
            R = reg * traceG;
        else
            R = reg;
        end
        G = G + R * eye(n_neighbors);
        w = G \ v;
        B(i,:) = (w / sum(w))';
    end
end

