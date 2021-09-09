% Sample calling syntax for testing taopounders and comparing to fminsearch
% ProblemFinalize is called after solving the instance

% Pad the history if there are remaining evaluations or truncate if too many
if nfev < nfmax
  fvals = [fvals(1:nfev);ones(nfmax-nfev,1)*fvals(nfev)];
else
  fvals = fvals(1:nfmax);
  fvecs = fvecs(1:nfmax,:);
  X_hist = X_hist(1:nfmax,:);
end

% Store the history and results for taopounders
SolverNumber = 1;
Results{SolverNumber,np}.alg = 'TAO Pounders';
Results{SolverNumber,np}.problem = ['problem ' num2str(np) ' from More/Wild'];
Results{SolverNumber,np}.H = fvals;
Results{SolverNumber,np}.X = X_hist;
Results{SolverNumber,np}.fvecs = fvecs;

% Initialize the function handle for evaluating the norm of the residuals
func = @(x)dfovec_wrap(m,n,x,nprob,0);

% Initialize the algorithmic parameters for fminsearch
rand('seed',0);
options = optimset('MaxFunEvals',nfmax,'MaxIter',nfmax);

% Reset the global history of the evaluations
nfev = 0;
fvals = zeros(nfmax,1);
fvecs = zeros(nfmax,m);
X_hist = zeros(nfmax,n);

% Call fminsearch
fminsearch(func,X0,options);

% Pad the history if there are remaining evaluations or truncate if too many
if nfev < nfmax
  fvals = [fvals(1:nfev);ones(nfmax-nfev,1)*fvals(nfev)];
else
  fvals = fvals(1:nfmax);
  fvecs = fvecs(1:nfmax,:);
  X_hist = X_hist(1:nfmax,:);
end

% Store the history and results for taopounders
SolverNumber = 2;
Results{SolverNumber,np}.alg = 'fminsearch';
Results{SolverNumber,np}.problem = ['problem ' num2str(np) ' from More/Wild'];
Results{SolverNumber,np}.H = fvals;
Results{SolverNumber,np}.X = X_hist;
Results{SolverNumber,np}.fvecs = fvecs;

