classdef PT_Bayes < handle
    % Problem Transformation using Naive Bayes
    properties
        model
        train_X
        train_Y
        toSLFcn = @LDL2SL;
    end
    methods
        function obj = PT_Bayes(train_X, train_Y, toSL)
            if nargin>=3 && ~isempty(toSL)
                obj.toSLFcn = toSL;
            end
            obj.train_X = train_X;
            obj.train_Y = obj.toSLFcn(train_Y);
        end
        function fit(obj)
            % MATLAB ClassificationNaiveBayes requires categorical labels 1..K
            labels = obj.train_Y + 1;
            obj.model = fitcnb(obj.train_X, labels);
        end
        function Yhat = predict(obj, X)
            [~, post] = predict(obj.model, X);
            Yhat = post;
        end
    end
end

function Yhat = LDL2SL(Y)
% Sample a single label from distribution per row
    n = size(Y,1);
    Yhat = zeros(n,1);
    for i=1:n
        p = Y(i,:);
        % mnrnd returns one-hot
        onehot = mnrnd(1, p);
        [~, idx] = max(onehot);
        Yhat(i) = idx-1; % align with Python 0-based if needed
    end
end

