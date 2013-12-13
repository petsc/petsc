

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ldltest.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test file for ldldown.m, ldlup.m, and (called by these) ldlrk1.m

clear;
n=5;

% generate random unit lower triangular matrix
L=tril(rand(n));L=L-diag(diag(L))+eye(n);

% random column
d=rand(n,1);

for j=1:n,
  disp('***** downdate *****')
  [L,d]=ldldown(L,d,j);
  disp('***** update *****')
  g=rand(n,1);
  g(j)=1; % encourage definiteness of update
  [L,d,p]=ldlup(L,d,j,g);  
end;
