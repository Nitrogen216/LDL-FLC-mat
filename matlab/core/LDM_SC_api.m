function varargout = LDM_SC_api(action, varargin)
switch action
    case 'bipart'
        [varargout{1:nargout}] = bipart(varargin{:});
    case 'solve'
        [varargout{1:nargout}] = solve(varargin{:});
    otherwise
        error('Unknown action');
end
end

function [Z, loss] = LDM_loss(D, P)
    if nargin >=2 && ~isempty(P)
        D = D .* P(:);
    end
    Z = barycenter_kneighbors_graph(D'); Z = Z';
    loss = sum((D - D*Z).^2, 'all');
end

function [P] = solve_P(D, Z0, Z1, rho, l)
    I = eye(size(D,2));
    IZ0 = I - Z0; IZ1 = I - Z1;
    one_vec = ones(size(D,2),1);
    function [f,g] = obj(Pvec)
        Pvec = Pvec(:);
        if l==0
            loss1 = 0; grad1 = 0;
        else
            Ph = abs(Pvec - 0.5);
            sgn = sign(0.5 - Pvec);
            ind = Ph < rho;
            grad1 = sgn .* double(ind);
            loss1 = sum(rho - Ph(ind));
        end
        DP0 = D .* (Pvec*one_vec');
        DP1 = D - DP0;
        DPZ0 = DP0 * IZ0; DPZ1 = DP1 * IZ1;
        loss = 0.5*(sum(DPZ0.^2, 'all') + sum(DPZ1.^2, 'all'));
        grad = (D .* (DPZ0*IZ0' - DPZ1*IZ1')) * one_vec;
        f = loss + l*loss1;
        g = grad + l*grad1;
    end
    n = size(D,1);
    P0 = rand(n,1);
    lb = zeros(n,1); ub = ones(n,1);
    opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','off','Algorithm','interior-point','MaxIterations',300);
    P = fmincon(@obj, P0, [],[],[],[], lb, ub, [], opts);
    P = P(:);
end

function [losses, Pvec] = bipart(D, inds, rho, l, iters)
    if nargin<5, iters=100; end
    losses = [];
    P = rand(size(D,1),1);
    [~, loss0] = LDM_loss(D, P);
    losses(end+1) = loss0; %#ok<AGROW>
    for i=1:iters
        [Z0,Z1] = solve_Z(D, P);
        P = solve_P(D, Z0, Z1, rho, l);
        [~, loss_i] = LDM_loss(D, P);
        losses(end+1) = loss_i; %#ok<AGROW>
    end
    Pvec = P(:);
end

function [Z0, Z1] = solve_Z(D, P)
    DP0 = D .* P(:);
    DP1 = D - DP0;
    Z0 = barycenter_kneighbors_graph(DP0'); Z0 = Z0';
    Z1 = barycenter_kneighbors_graph(DP1'); Z1 = Z1';
end

function [cluster_labels, manifolds] = solve(Y, r, rho, l)
    if nargin<2, r=100; end
    if nargin<3, rho=0.1; end
    if nargin<4, l=1; end
    clusters = {};
    manifolds = {};
    n = size(Y,1);
    function partition(inds)
        D = Y(inds,:);
        [Z, loss] = LDM_loss(D);
        if numel(inds) < r
            clusters{end+1} = inds; %#ok<AGROW>
            manifolds{end+1} = Z; %#ok<AGROW>
            return;
        end
        [~, P] = bipart(D, inds, rho, l);
        [~, Zfull] = LDM_loss(D);
        [loss0, ~] = LDM_loss(D, P);
        [loss1, ~] = LDM_loss(D, 1-P);
        if loss <= loss0 + loss1
            clusters{end+1} = inds; %#ok<AGROW>
            manifolds{end+1} = Z; %#ok<AGROW>
            return;
        end
        inds0 = inds(P>0.5);
        inds1 = inds(P<=0.5);
        if isempty(inds0) || isempty(inds1)
            clusters{end+1} = inds; %#ok<AGROW>
            manifolds{end+1} = Z; %#ok<AGROW>
            return;
        end
        partition(inds0);
        partition(inds1);
    end
    partition((1:n)');
    cluster_labels = zeros(n,1);
    for j=1:numel(clusters)
        cluster_labels(clusters{j}) = j;
    end
end

