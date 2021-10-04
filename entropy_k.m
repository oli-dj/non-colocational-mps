function [H,I] = entropy_k(P)
%ENTROPY_K Calculate the information content and entropy of soft data
%   Supports any k number of categories/facies
%   Adapted from Hansen, T.M et al. 2018
%
%   Oli D. Johannsson, oli@johannsson.dk

% Find the length of the probability mass function P(x)
k = length(P);
I = zeros(size(P));

% Find nonzero elements of P(X)
idx = P>0;

% Calculate information content of X
%   I(X) = log_k(P(X)), log_k(x) = ln(x)/ln(k)
I(idx) = -log(P(idx))/log(k);    

% Calculate the Entropy H(X)
H = sum(P(idx).*I(idx));
end