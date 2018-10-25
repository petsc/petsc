% Sample calling syntax for testing taopounders and comparing to fminsearch
% ProblemInitialize is called prior to solving the instance
%   The taopounders driver sets np, the problem instance number
%   This code then initializes the rest of the information needed

nprob = dfo(np,1);    % Internal index for the problem
n = dfo(np,2);        % Number of variables
m = dfo(np,3);        % Number of residuals

% Initialize the starting point
factor = 10;
factor_power = dfo(np,4);
X0 = dfoxs(n,nprob,factor^factor_power)';

% Initialize the function handle for evaluating the residuals
func = @(x)dfovec_wrap(m,n,x,nprob,1);
jac = @(x)jacobian(m,n,x,nprob);

% Initialize the algorithmic parameters for taopounders
nfmax = nf_const*(n+1);   % Maximum number of function evaluations
npmax = 2*n+1;            % Maximum number of interpolation points
delta = 0.1;              % Initial trust region radius

% Reset the global history of the evaluations
nfev = 0;
fvals = zeros(nfmax,1);
fvecs = zeros(nfmax,m);
X_hist = zeros(nfmax,n);

% Control returns to taopounders interface to solve the problem

