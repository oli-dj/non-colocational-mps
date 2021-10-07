function [path, n_u] = pref_path(SG,SDG,I_fac)
%PREF_PATH Preferential random path
%   As defined in Hansen, T.M. et al. (2018)
%   Uses entropy_k.m to calculate the entropy.
%
% SG        Simulation Grid (x by y by z) (with or w/o hard data)
% SDG       Soft Data Grid (x by y by z by k)
% I_fac     Randomness factor
%
% Oli D. Johannsson, 2018

dim = ndims(SG);

%Uninformed nodes
uninformed = isnan(SG);

% Number of uninformed nodes
n_u = sum(sum(sum(uninformed)));

% Create raster path
switch dim
    case 2 %2D
        % Fill NaN's
        SDG(isnan(SDG))=1./size(SDG,dim+1);
        % Size of simulation grid
        [nx, ny] = size(SG);
        X=1:nx;
        Y=1:ny;
        [XX,YY]=meshgrid(X,Y);
        
        path = [XX(uninformed(1:end))',YY(uninformed(1:end))'];
        
    case 3 %3D
        % Fill NaN's
        SDG(isnan(SDG))=1./size(SDG,dim+1);
        % Size of simulation grid
        [nx, ny, nz] = size(SG);
        X=1:nx;
        Y=1:ny;
        Z=1:nz;
        [XX,YY,ZZ]=meshgrid(X,Y,Z);
        
        path = [XX(uninformed(1:end))',YY(uninformed(1:end))',...
            ZZ(uninformed(1:end))'];
end

% Calculate entropy at each uninformed node
H = zeros(n_u,1);
switch dim
    case 2
        for i = 1:n_u
            H(i) = entropy_k(SDG(path(i,1),path(i,2),:));
        end
    case 3
        for i = 1:n_u
            H(i) = entropy_k(SDG(path(i,1),path(i,2),path(i,3),:));
        end
end

% Certainty
C = 1 - H./max(H);

% Generate random preferential order
order = rand(n_u,1) - 1 + I_fac.*C;
[~,i] = sort(order);

% Sort path by the random preferential order
path = path(i,:);

end