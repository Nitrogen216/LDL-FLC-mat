classdef LDL_LRR < handle
    % MATLAB port of ldllrr.py using fminunc
    properties
        lam = 1e-3
        beta = 1
        random_state = 123
        W
        b % bias vector
    end
    methods
        function obj = LDL_LRR(varargin)
            if ~isempty(varargin)
                for i=1:2:numel(varargin)
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
        end
        function self = fit(self, X, D)
            rng(self.random_state);
            [n,d] = size(X); L = size(D,2);
            % parameters: W (d x L), b (1 x L)
            W0 = randn(d,L)*0.01; b0 = zeros(1,L);
            theta0 = [W0(:); b0(:)];
            P = sigmoid(D - permute(D,[1 3 2])); % N x L x L
            P = double(P > 0.5);
            Wpair = (D - permute(D,[1 3 2])).^2; % weights
            function [f,g] = obj(theta)
                W = reshape(theta(1:d*L), d, L);
                b = reshape(theta(d*L+1:end), 1, L);
                Z = X*W + b;
                Dhat = softmax(Z);
                klloss = -sum(D .* log(Dhat+1e-9), 'all');
                % ranking loss
                Phat = sigmoid((Dhat - permute(Dhat,[1 3 2])) * 100); % N x L x L
                Phat = max(min(Phat,1-1e-9),1e-9);
                lrank = ((1-P) .* log(1-Phat) + P .* log(Phat)) .* Wpair;
                rankloss = -sum(lrank, 'all');
                reg = sum(W(:).^2) + sum(b(:).^2);
                f = self.lam/(2*n) * rankloss + klloss + self.beta/2 * reg;
                if nargout>1
                    % gradients
                    % d klloss / dZ = Dhat - D
                    dZ = (Dhat - D);
                    % rank gradient w.r.t. Dhat via chain rule
                    Sig = Phat; P0 = P; Wp = Wpair;
                    dPhat = (100) * Sig .* (1-Sig); % d sigmoid
                    % d rankloss / d Dhat
                    G = zeros(n,L);
                    for l1=1:L
                        for l2=1:L
                            dL_dPh = -((1-P0(:,l1,l2))./(1-Sig(:,l1,l2)) - P0(:,l1,l2)./Sig(:,l1,l2)) .* Wp(:,l1,l2);
                            dL_dDhat_l1 = dL_dPh .* dPhat(:,l1,l2);
                            dL_dDhat_l2 = -dL_dPh .* dPhat(:,l1,l2);
                            G(:,l1) = G(:,l1) + dL_dDhat_l1;
                            G(:,l2) = G(:,l2) + dL_dDhat_l2;
                        end
                    end
                    dZ = dZ + self.lam/(2*n) * G;
                    % backprop to W,b
                    gW = X' * dZ + self.beta * W;
                    gb = sum(dZ,1) + self.beta * b;
                    g = [gW(:); gb(:)];
                end
            end
            opts = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'Display','off','MaxIterations',500);
            theta = fminunc(@obj, theta0, opts);
            self.W = reshape(theta(1:d*L), d, L);
            self.b = reshape(theta(d*L+1:end), 1, L);
        end
        function Dhat = predict(self, X)
            Z = X * self.W + self.b;
            Dhat = softmax(Z);
        end
    end
end

function S = sigmoid(X)
    S = 1 ./ (1 + exp(-X));
end

function Y = softmax(Z)
    Z = Z - max(Z, [], 2);
    E = exp(Z);
    S = sum(E, 2);
    Y = E ./ S;
end

