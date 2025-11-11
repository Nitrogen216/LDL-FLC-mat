classdef AA_KNN < handle
    % Adaptive Algorithm K-Nearest Neighbors for LDL
    properties
        k = 1;
        train_X
        train_Y
    end
    methods
        function obj = AA_KNN(k)
            if nargin>0, obj.k = k; end
        end
        function fit(obj, X, Y)
            obj.train_X = X; obj.train_Y = Y;
        end
        function Yhat = predict(obj, X)
            n = size(X,1);
            Yhat = zeros(n, size(obj.train_Y,2));
            if exist('knnsearch','file') == 2
                [idx, ~] = knnsearch(obj.train_X, X, 'K', obj.k);
                for i=1:n
                    Yhat(i,:) = mean(obj.train_Y(idx(i,:),:),1);
                end
            else
                % brute-force distances
                for i=1:n
                    diffs = obj.train_X - X(i,:);
                    d2 = sum(diffs.^2,2);
                    [~, order] = sort(d2, 'ascend');
                    nn = order(1:obj.k);
                    Yhat(i,:) = mean(obj.train_Y(nn,:),1);
                end
            end
        end
        function s = char(obj)
            s = sprintf('AA_KNN_K=%d', obj.k);
        end
    end
end

