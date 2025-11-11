function Yhat = LDL2SL(Y)
% Convert label distribution to single label by sampling
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

