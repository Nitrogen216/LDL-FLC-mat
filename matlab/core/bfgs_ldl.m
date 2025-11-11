classdef bfgs_ldl < handle
    % BFGS-based Label Distribution Learning
    properties
        C = 0;
        W
        x
        y
        n_features
        n_outputs
    end
    methods
        function obj = bfgs_ldl(C)
            if nargin>0, obj.C = C; end
        end
        function fit(obj, train_x, train_y)
            obj.x = train_x; obj.y = train_y;
            obj.n_features = size(train_x,2);
            obj.n_outputs = size(train_y,2);
            w0 = rand(obj.n_features * obj.n_outputs,1)*0.2-0.1;
            opts = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'Display','off','MaxIterations',300);
            fun = @(w) obj.object_fun(w);
            w = fminunc(fun, w0, opts);
            obj.W = reshape(w, obj.n_outputs, obj.n_features)';
        end
        function [loss, gradvec] = object_fun(obj, weights)
            W = reshape(weights, obj.n_outputs, obj.n_features)';
            p = softmax(obj.x * W);
            y_true = min(max(obj.y, 1e-7), 1);
            p = min(max(p, 1e-7), 1);
            func_loss = -sum(y_true .* log(p), 'all') + obj.C * 0.5 * (weights' * weights);
            gradM = obj.x' * (p - obj.y);
            gradM = gradM + obj.C * W;
            gradvec = gradM(:);
            loss = func_loss;
        end
        function p = predict(obj, x)
            p = softmax(x * obj.W);
        end
        function s = char(obj)
            s = sprintf('bfgs_ldl_%.4g', obj.C);
        end
    end
end

function Y = softmax(Z)
    Z = Z - max(Z, [], 2);
    E = exp(Z);
    S = sum(E, 2);
    Y = E ./ S;
end

