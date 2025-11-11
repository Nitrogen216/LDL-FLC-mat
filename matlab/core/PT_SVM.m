classdef PT_SVM < handle
    % Problem Transformation using SVM
    properties
        model
        C = 1;
        train_X
        train_Y
        toSLFcn = @LDL2Bayes;
    end
    methods
        function obj = PT_SVM(train_X, train_Y, C, toSL)
            if nargin>=3 && ~isempty(C), obj.C = C; end
            if nargin>=4 && ~isempty(toSL), obj.toSLFcn = toSL; end
            obj.train_X = train_X;
            obj.train_Y = obj.toSLFcn(train_Y);
        end
        function fit(obj)
            labels = obj.train_Y + 1;
            t = templateSVM('KernelFunction','rbf','KernelScale','auto','BoxConstraint',obj.C);
            obj.model = fitcecoc(obj.train_X, labels, 'Learners', t);
        end
        function Yhat = predict(obj, X)
            Yhat = predict(obj.model, X);
            Yhat = Yhat - 1; % 0-based to match python
        end
        function s = char(obj)
            s = sprintf('PT-SVM_C=%.4g', obj.C);
        end
    end
end

function Yhat = LDL2Bayes(Y)
    [~, Yhat] = max(Y, [], 2);
    Yhat = Yhat - 1; % 0-based
end

