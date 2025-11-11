function Yhat = LDL2Bayes(Y)
% Convert label distribution to single label by taking argmax
    [~, Yhat] = max(Y, [], 2);
    Yhat = Yhat - 1; % 0-based
end

