function varargout = ldl_metrics(action, varargin)
% Metrics: call as [cheby, clark, can, kl, cosine, inter] = score(Y, Yhat)
switch action
    case 'score'
        [varargout{1:nargout}] = score(varargin{:});
    otherwise
        error('Unknown action %s', action);
end
end

function Yp = proj(Y)
% Project rows onto probability simplex
    [n, m] = size(Y);
    X = sort(Y, 2, 'descend');
    Xtmp = (cumsum(X,2) - 1) .* (1 ./ (1:m));
    rho = sum(X > Xtmp, 2);
    theta = Xtmp(sub2ind(size(Xtmp), (1:n)', rho));
    Yp = max(Y - theta, 0);
end

function v = KL_div(Y, Yhat)
    epsv = eps;
    Y = max(min(Y,1), epsv);
    Yhat = max(min(Yhat,1), epsv);
    kl = sum(Y .* (log(Y) - log(Yhat)), 2);
    v = mean(kl);
end

function v = Cheby(Y, Yhat)
    v = mean(max(abs(Y - Yhat), [], 2));
end

function v = Clark(Y, Yhat)
    epsv = eps;
    Y = max(min(Y,1), epsv);
    Yhat = max(min(Yhat,1), epsv);
    v = mean(sqrt(sum(((Y - Yhat).^2) ./ ((Y + Yhat).^2), 2)));
end

function v = Canberra(Y, Yhat)
    epsv = eps;
    Y = max(min(Y,1), epsv);
    Yhat = max(min(Yhat,1), epsv);
    v = mean(sum(abs(Y - Yhat) ./ (Y + Yhat), 2));
end

function v = Cosine(Y, Yhat)
    s = sum(Y .* Yhat, 2);
    m = vecnorm(Y,2,2) .* vecnorm(Yhat,2,2);
    v = mean(s ./ m);
end

function v = Intersection(Y, Yhat)
    l1 = sum(abs(Y - Yhat), 2);
    v = 1 - 0.5 * mean(l1);
end

function varargout = score(Y, Yhat)
    cheby = Cheby(Y, Yhat);
    clark = Clark(Y, Yhat);
    can = Canberra(Y, Yhat);
    kl = KL_div(Y, Yhat);
    cosine = Cosine(Y, Yhat);
    inter = Intersection(Y, Yhat);
    varargout = {cheby, clark, can, kl, cosine, inter};
end

