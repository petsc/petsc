global fvals fvecs nfev X_hist 
addpath('more_wild_probs/')

% gtol    [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
gtol = 1e-13;
% delta   [dbl] Positive trust region radius (.1)
delta = .1;
% nfs  [int] Number of function values (at X0) known in advance (0)
nfs = 0;
% F0      [dbl] [fstart-by-1] Set of known function values  ([])
F0 = [];
% xind    [int] Index of point in X0 at which to start from (1)
xind = 1;
% Low     [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
% Upp     [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
% printf  [log] 1 Indicates you want output to screen (1)
printf = false;

% The following are problem specific
% func    [f h] Function handle so that func(x) evaluates f (@calfun)    
% n       [int] Dimension (number of continuous variables)
% X0      [dbl] [min(fstart,1)-by-n] Set of initial points  (zeros(1,n))
% nfmax   [int] Maximum number of function evaluations (>n+1) (100)
% m [int] number of residuals
% npmax   [int] Maximum number of interpolation points (>n+1) (2*n+1)

factor = 10;

load dfo.dat
probtype = 'smooth';
to_solve = 1:53;
nf_const = 10;

SolverNumber = 2;

load results Results;
H = inf(nf_const*(max(dfo(to_solve))+1),length(to_solve),SolverNumber);
for np = to_solve
    for s = 1:SolverNumber
        H(1:length(Results{s,np}.H),np,s) = Results{s,np}.H;     
    end
end
h = perf_profile(H,1e-3,0);
legend(h,{Results{1,1}.alg, Results{2,1}.alg});
saveas(gca,'perf.png');

