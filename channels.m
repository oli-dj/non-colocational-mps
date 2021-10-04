% channels : 2D [250x250] training image
% from Strebelle (2000)
% implemented in mGstat: https://github.com/cultpenguin/mGstat
function d=channels
d=load('channels.mat');
d=d.channels;
