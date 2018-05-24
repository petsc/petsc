if length(fvals) < nfmax
  fvals = [fvals;ones(nfmax-length(fvals),1)*fvals(nfev)];
else
  fvals = fvals(1:nfmax);
  X_hist = X_hist(1:nfmax,:);
  fvecs = fvecs(1:nfmax,:);
end
SolverNumber = SolverNumber + 1;
    
Results{SolverNumber,np}.alg = 'TAO Pounders';
Results{SolverNumber,np}.problem = ['problem ' num2str(np) ' from More/Wild'];
Results{SolverNumber,np}.H = fvals;
Results{SolverNumber,np}.X = X_hist;
Results{SolverNumber,np}.fvecs = fvecs;
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fminsearch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_hist = zeros(nfmax,n);
nfev = 0;
fvals = zeros(2,1);
fvecs = zeros(nfmax,m);       
func = @(x)dfovec_wrap(m,n,x,nprob,0);

rand('seed',0);
options = optimset('MaxFunEvals',nfmax,'MaxIter',nfmax);
[xbest] = fminsearch(func,xs, options);
    
if length(fvals) < nfmax
  fvals = [fvals;ones(nfmax-length(fvals),1)*fvals(nfev)];
else
  fvals = fvals(1:nfmax);
  X_hist = X_hist(1:nfmax,:);
  fvecs = fvecs(1:nfmax,:);
end
SolverNumber = SolverNumber + 1;
Results{SolverNumber,np}.alg = 'fminsearch';
Results{SolverNumber,np}.problem = ['problem ' num2str(np) ' from More/Wild'];
Results{SolverNumber,np}.H = fvals;
Results{SolverNumber,np}.X = X_hist;
Results{SolverNumber,np}.fvecs = fvecs;

