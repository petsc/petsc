nprob = dfo(np,1);
n = dfo(np,2);
m = dfo(np,3);
factor_power = dfo(np,4);
xs = dfoxs(n,nprob,factor^factor_power)';    
nfmax = nf_const*(n+1);
SolverNumber = 0;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize Tao Pounders
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
X_hist = zeros(nfmax,n);                         
nfev = 0;
fvals = zeros(2,1);            
fvecs = zeros(nfmax,m); 
func = @(x)dfovec_wrap(m,n,x,nprob,1);

npmax = 2*n+1;
Low = -Inf(1,n);
Upp = Inf(1,n);
 
X = xs;

