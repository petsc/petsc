function out = dfovec_wrap(m,n,x,nprob,vec_out)

global nfev fvals fvecs X_hist 

fvec = dfovec(m,n,x,nprob);
y = fvec'*fvec;

% Update the function value history
nfev = nfev+1;
fvecs(nfev,:) = fvec;
fvals(nfev,:) = y;
X_hist(nfev,:) = x';

if vec_out
  out = fvec;
else
  out = y;
end
