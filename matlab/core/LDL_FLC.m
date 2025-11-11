classdef LDL_FLC < handle
    properties
        g
        l1
        l2
        train_x
        train_y
        U
        manifolds
        n_features
        n_outputs
        W
        J
    end
    methods
        function obj = LDL_FLC(g, l1, l2)
            obj.g = g; obj.l1 = l1; obj.l2 = l2;
        end
        function fit(obj, x, y, U, manifolds)
            obj.train_x = x; obj.train_y = y;
            if nargin < 4 || isempty(U)
                obj.U = fuzzy_cmeans(y, obj.g);
            else
                obj.U = U;
            end
            if nargin < 5 || isempty(manifolds)
                obj.manifolds = solve_LDM(y, obj.U);
            else
                obj.manifolds = manifolds;
            end
            obj.n_features = size(obj.train_x,2);
            obj.n_outputs = size(obj.train_y,2);
            obj.J = ones(obj.n_outputs);
        end
        function [loss, gradvec] = objective_func(obj, Wvec)
            W = reshape(Wvec, obj.n_features, obj.n_outputs);
            y_hat = softmax(obj.train_x * W);
            [loss, grad] = KL(obj.train_y, y_hat);
            if obj.l2 ~= 0
                for j=1:obj.g
                    [ldm_loss, ldm_grad] = obj.LDM(y_hat, obj.U(:,j), obj.manifolds{j});
                    loss = loss + obj.l2 * ldm_loss;
                    grad = grad + obj.l2 * ldm_grad;
                end
            end
            grad = obj.train_x' * grad;
            if obj.l1 ~= 0
                loss = loss + 0.5 * obj.l1 * sum(W(:).^2);
                grad = grad + obj.l1 * W;
            end
            gradvec = grad(:);
        end
        function [loss, gradient] = LDM(obj, y_hat, u, I_Z)
            u = u(:);
            IZ = I_Z;
            DP = y_hat * IZ;  % n x m
            loss = 0.5 * sum((u .* DP).^2, 'all');
            H = (u .* (u .* y_hat)) * (IZ * IZ');
            gradient = (H - (H .* y_hat) * obj.J) .* y_hat;
        end
        function solve(obj, max_iters)
            if nargin < 2, max_iters = 600; end
            W0 = eye(obj.n_features, obj.n_outputs);
            W0 = W0(:);
            opts = optimoptions('fminunc', 'Algorithm','quasi-newton', ...
                'SpecifyObjectiveGradient', true, 'Display','off', 'MaxIterations', max_iters);
            fun = @(w) obj.objective_func(w);
            Wopt = fminunc(fun, W0, opts);
            obj.W = reshape(Wopt, obj.n_features, obj.n_outputs);
        end
        function Yhat = predict(obj, X)
            Yhat = softmax(X * obj.W);
        end
        function s = char(obj)
            s = sprintf('LDLFLC_%d_%.4g_%.4g', obj.g, obj.l1, obj.l2);
        end
    end
end

function [loss, grad] = KL(y, y_hat)
    epsv = 1e-15;
    y = max(min(y,1), epsv);
    y_hat = max(min(y_hat,1), epsv);
    loss = -sum(y .* log(y_hat), 'all');
    grad = (y_hat - y);
end

function Y = softmax(Z)
    Z = Z - max(Z, [], 2);
    E = exp(Z);
    S = sum(E, 2);
    Y = E ./ S;
end

function Z = solve_Z(D, P)
    if isvector(P), P = P(:); end
    DP = D .* P;  % n x m
    Z = barycenter_kneighbors_graph(DP');
    Z = Z';
end

function manifolds = solve_LDM(D, U)
    manifolds = cell(1, size(U,2));
    I = eye(size(D,2));
    for j=1:size(U,2)
        manifolds{j} = I - solve_Z(D, U(:,j));
    end
end

function U = fuzzy_cmeans(data, g)
    % Use Fuzzy Logic Toolbox if available; otherwise, simple FCM fallback
    if exist('fcm','file') == 2
        [~, Umat] = fcm(data, g, [2 1000 1e-3 false]); % dim x N membership
        U = Umat'; % N x g
        return;
    end
    % Fallback FCM (m=2)
    m = 2; maxiter = 1000; tol = 1e-3;
    N = size(data,1);
    % init random U
    U = rand(N, g); U = U ./ sum(U,2);
    for it=1:maxiter
        % update centers
        um = U.^m;
        centers = (um' * data) ./ sum(um,1)'; % g x d
        % update U
        dist = zeros(N, g);
        for j=1:g
            diff = data - centers(j,:);
            dist(:,j) = sqrt(sum(diff.^2,2)) + 1e-12;
        end
        U_new = zeros(N,g);
        for j=1:g
            U_new(:,j) = 1 ./ sum( (dist(:,j) ./ dist) .^ (2/(m-1)), 2 );
        end
        if sqrt(sum((U_new - U).^2, 'all')) < tol, U = U_new; break; end
        U = U_new;
    end
end

