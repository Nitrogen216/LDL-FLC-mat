classdef LDLLDM_Cluster < handle
    % Helper class for LDLLDM_Full
    % Represents a cluster with its own label distribution manifold
    properties
        X
        l
        I
        Z
        I_Z
        L
    end
    
    methods
        function obj = LDLLDM_Cluster(X, l, Y, Z)
            obj.X = X;
            obj.l = l;
            obj.I = eye(size(Y, 2));
            obj.LDM(Y, Z);
        end
        
        function LDM(obj, Y, Z)
            if isempty(Z)
                obj.Z = barycenter_kneighbors_graph(Y');
                obj.Z = obj.Z';
            else
                obj.Z = Z;
            end
            obj.I_Z = obj.I - obj.Z;
            obj.L = obj.I_Z * obj.I_Z';
        end
        
        function [loss, grad] = LDL(obj, Y_hat)
            if obj.l == 0
                loss = 0;
                grad = 0;
                return;
            end
            
            YIZ = Y_hat * obj.I_Z;
            loss = sum(YIZ(:).^2);
            
            % weighted_ME gradient
            W = 2 * (Y_hat * obj.L);
            grad = obj.X' * weighted_ME(Y_hat, W);
            
            loss = obj.l * loss;
            grad = obj.l * grad;
        end
    end
end

function grad = weighted_ME(Y_hat, W)
    % Compute weighted maximum entropy gradient
    weighted_Y_hat = Y_hat .* W;
    grad = -sum(weighted_Y_hat, 2) .* Y_hat + weighted_Y_hat;
end

