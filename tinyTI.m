function [ TI ] = tinyTI()
%TINYTI Summary of this function goes here
%   Training image from Strubhaar et al 2011
    
%     TI = [  0   0   1   0   1   1;
%             0   1   1   1   1   0;
%             0   1   0   0   0   0;
%             1   1   1   0   1   1;
%             0   0   1   1   1   0;
%             0   1   1   0   0   0];

   TI = [1 1 0 1 0 0
    1 0 0 0 0 1
    1 0 1 1 1 1
    0 0 0 1 0 0
    1 1 0 0 0 1
    1 0 0 1 1 1
];
        
end

