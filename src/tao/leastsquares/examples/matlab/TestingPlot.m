% Sample calling syntax for testing taopounders and comparing to fminsearch
% TestingPlot is called after taopounders completes.  It reads the results 
% from results.mat and produces a performance profile comparing
% taopounders to fminsearch.

% Make the test problem accessible to the matlab session and load the data matrix
addpath('more_wild_probs/')
load dfo.dat

% Initialize the problems that are being solved and retrieve the results
to_solve = 1:53;
load results Results;

% Initialize the constant factor for the number of allowable function evaluations
nf_const = 10;

% Produce the performance profile and save the plot for the two solvers
SolverNumber = 2;
H = inf(nf_const*(max(dfo(to_solve,2))+1),length(to_solve),SolverNumber);
for np = to_solve
    for s = 1:SolverNumber
        H(1:length(Results{s,np}.H),np,s) = Results{s,np}.H;
    end
end
h = perf_profile(H,1e-3,0);
legend(h,{Results{1,1}.alg, Results{2,1}.alg});
saveas(gca,'perf.png');

