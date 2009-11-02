function Fvec = nls_f(X)
% This routine is used to specify a generic function called by mnhnls
% Takes n-by-1 column vector X and outputs m-by-1 vector Fvec.

global m nfev Fvals Xhist Fhist Deltahist delta % Global vars used in mnhnls_driver
n = length(X); % Problem dimension 
nfev = nfev +1;

dat = dlmread('chwirut.dat');
y = dat[:,1];
t = dat[:,2];
m = zeros(size(y),1);
Fvec=ones(m);
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Generate the vector through internal or external routine between the ++++
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Send X to file:

for i=1:m
  Fvec(i)= y(i) - exp(-X(1)*t(i)) / (X(2) + X(3)*t(i));
end

% [OPTIONAL] CORRECTION DONE TO AVOID HUGE VALUES:
Fvec = max(Fvec,-1e64);
Fvec = min(Fvec,1e64);

% Save details for the final output:
Fvals(nfev) = sum(Fvec.^2); % Stores the history of the objective vals
size(X)
size(Xhist)
Fhist(nfev,:) = Fvec';      % Stores the history of the fvec vals
Xhist(nfev,:) = X;         % Stores the history of the x vals
Deltahist(nfev) = delta;
