% README for mfq
% Updated 12/12/2007 by Stefan WIld

% D       [np-by-n double] Stores the evaluation point displacements
% F       [nfmax-by-1 double] Stores the function values of evaluated points
% G       [double] n-by-1 model gradient at Xk
% GPoints [n-by-n double]    Displacements for geometry improving points
% H       [double] n-by-n model Hessian at Xk
% Modeld  [1-by-n double]    Unit direction to improve model
% ModelIn [npmax-by-1 integer]   integer vector of model interpolation indices
% X       [nfmax-by-n double] Stores the evaluation point locations
% X0      [double]  1-by-n initial point
% Xsp     [1-by-n double] subproblem solution
% delta   [double]    Trust region radius (>0)
% beta    [double]   Trust region parameter for the model gradient
% c       [double] model value at Xk
% c1      [double]    Factor for checking validity
% mdec    [double]    Decrease predicted by the model
% n       [integer]   dimension (number of continuous variables)
% nf      [integer]   Counter for the number of function evaluations
% ng      [double]   dimension (number of continuous variables)
% np      [integer]   Number of model interpolation points
% nfmax   [integer]   Maximum number of function evaluations (assumed >n+1)
% npmax   [integer]   Maximum number of model interpolation points (assumed >n+1)
% rho     [double]    ratio of actual decrease to model decrease
% xkin    [integer]   index of current center
% valid   [logical]   flag saying if model is valid within C1*delta
% gradtol [double]    Tolerance for the 2-norm of the model gradient (1e-4)
% maxdelta[double]    Maximum trust region radius (>=delta)
% mindelta[double]    Minimum tr radius (technically must be 0)
% theta1  [double]    Pivot threshold for validity
% theta2  [double]    Pivot threshold for additional points
% gam0    [double]    Parameter for shrinking delta (> 1)
% gam1    [double]    Parameter for enlarging delta (> 1)
% eta0    [double]    Parameter 1 for accepting point (0<=eta0<eta1) (0)
% eta1    [double]    Parameter 2 for accepting point (eta0<eta1<1) (.8)
% rtol    [double]    Parameter used by gqt (0.001)
% itmax   [integer]   Parameter used by gqt (20)
% par     [double]    Parameter used by gqt (ng/delta)
