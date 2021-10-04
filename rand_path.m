function [path, n_u]=rand_path(SG)
%RAND_PATH Summary of this function goes here
%
%   Completely random path through the uninformed nodes in simulation grid
%   Supports 2D and 3D simulation grids.
%
%   outputs:
%       path:   random path
%       n_u:    number of uninformed nodes

%Uninformed nodes
uninformed = isnan(SG');

% Number of uninformed nodes
n_u = sum(sum(sum(uninformed)));

switch length(size(SG))
    case 2 %2D
        % Size of simulation grid
        [nx, ny] = size(SG);
        X=1:nx;
        Y=1:ny;
        [XX,YY]=meshgrid(X,Y);
        
        %create raster path
        path = [XX(uninformed(1:end))',YY(uninformed(1:end))'];
      
    case 3 %3D
        % Size of simulation grid
        [nx, ny, nz] = size(SG);
        X=1:nx;
        Y=1:ny;
        Z=1:nz;
        [XX,YY,ZZ]=meshgrid(X,Y,Z);
        
        %create raster path
        path = [XX(uninformed(1:end))',YY(uninformed(1:end))',...
            ZZ(uninformed(1:end))'];
end

%randomize path
path = path(randperm(n_u),:);