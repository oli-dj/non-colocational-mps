function [cmap] = generateColormap(SG,varargin)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Set max and min
if length(varargin) < 1
    minval = 0;
    maxval = 1;
else
    minval = varargin{1};
    maxval = varargin{2};
end

%Set number of steps
if length(varargin) > 2
    steps = varargin{3}
else
    steps = 128;
end

R = NaN(steps,1);
G = NaN(steps,1);
B = NaN(steps,1);

%Calculate mean
mu = mean(mean(mean(SG(~isnan(SG)))));

%Find index corresponding to mean value.
meanstep = round(steps * mu./(maxval - minval));

for i = 0:(meanstep - 2)
    R(i+1) = i./(steps./(steps./meanstep));
    G(i+1) = i./(steps./(steps./meanstep));
    B(i+1) = 1;
end

R(meanstep) = 1;
G(meanstep) = 1;
B(meanstep) = 1;

for i = 1:(steps-meanstep)
    R(meanstep+i) = 1;
    G(meanstep+i) = 1-i./(steps-meanstep);
    B(meanstep+i) = 1-i./(steps-meanstep);
    
    
end

cmap = [R,G,B];