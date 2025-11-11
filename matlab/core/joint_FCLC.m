function varargout = joint_FCLC(action, varargin)
switch action
    case 'get_fuzzy_manifolds'
        [varargout{1:nargout}] = get_fuzzy_manifolds(varargin{:});
    otherwise
        error('Unknown action %s', action);
end
end

function Z = solve_Z(D, P)
    if isvector(P), P = P(:); end
    DP = D .* P;
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

function U = update_membership(D, g, manifolds, m)
    if nargin < 4, m = 2; end
    % dis: N x g
    N = size(D,1);
    dis = zeros(N, g);
    for j=1:g
        IZ = manifolds{j};
        DP = D * IZ; % N x m
        dis(:,j) = sqrt(sum(DP.^2, 2));
    end
    U = zeros(N, g);
    for j=1:g
        U(:,j) = 1 ./ sum( (dis(:,j) ./ dis) .^ (2/(m-1)), 2 );
    end
end

function [U, manifolds] = joint_fc_ldm(D, g, max_iters, tol)
    if nargin < 3, max_iters = 150; end
    if nargin < 4, tol = 1e-3; end
    rng('default');
    U = rand(size(D,1), g);
    U = U ./ sum(U,2);
    for i=1:max_iters
        manifolds = solve_LDM(D, U);
        U_new = update_membership(D, g, manifolds);
        err = sqrt(sum((U_new - U).^2, 'all'));
        U = U_new;
        if err < tol, break; end
    end
end

function [U, manifolds] = get_fuzzy_manifolds(~, train_y, g)
    [U, manifolds] = joint_fc_ldm(train_y, g);
end

