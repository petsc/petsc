

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% MINQ    - bound constrained quadratic programming         %%%%%%
%%%%%%% MINQDEF - general definite quadratic programming          %%%%%%
%%%%%%% MINQSEP - definite separable quadratic programming        %%%%%%
%%%%%%% RLS     - robust least squares                            %%%%%%
%%%%%%%           using (sparse) rank 1 modifications             %%%%%%
%%%%%%% source: http://solon.cma.univie.ac.at/~neum/software/minq %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% MINQ minimizes an affine quadratic form subject to simple bounds,
% using coordinate searches and reduced subspace minimizations
%    min    fct = gamma + c^T x + 0.5 x^T G x 
%    s.t.   x in [xu,xo]    % xu<=xo is assumed
% where G is a symmetric n x n matrix, not necessarily definite
% (if G is indefinite, only a local minimum is found).
% If G is sparse, it is assumed that the ordering is such that
% a sparse modified Cholesky factorization is feasible
%
% MINQDEF minimizes a definite quadratic form 
% subject to linear constraints
%    min    fct = c^T x + 0.5 x^T G x 
%    s.t.   A x >= b, with equality at indices with eq=1
% where G is a definite symmetric n x n matrix.
% If A is sparse, it is assumed that the ordering is such that
% sparse Cholesky factorization of G and AG^(-1)A^T are feasible
%
% MINQSEP minimizes a definite separable quadratic form 
% subject to linear constraints
%    min    fct = c^T x + 0.5 x^T D x 
%    s.t.   A x >= b, with equality at indices with eq=1
% where D is a definite n x n diagonal matrix.
% If A is sparse, it is assumed that the ordering is such that
% a sparse Cholesky factorization of A^T is feasible
%
% RLS solves a linear least squares problem
%    min    ||Ax-b||_2^2  
%    s.t.   |x-x0|<=r
% If x0 and r have the default value, RLS can be used to
% regularize least squares problems. In ill-conditioned cases,
% it yields much better approximate solutions than x=A\b,
% though at a time penalty.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% minqdef.m		general definite QP solver   (calls minq.m)
% minqsep.m		definite separable QP solver (calls minq.m)
% minq.m		bound constrained QP solver
% minqsub.m		patch for minq.m containing the subspace search
% minq_test.m		small test program for minq.m
% rls.m			robust least squares solver
% rls_test.m		small test program for rls.m
% 
% minq.m calls the following subroutines:
% getalp.m		exact quadratic line search
% ldldown.m		LDL^T factorization downdate
% ldlrk1.m		LDL^T factorization rank 1 change
% ldlup.m		LDL^T factorization update
% ldltest.m		LDL^T factorization update
% pr01.m		print characteristic vecor of activities
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% minq.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [x,fct,ier]=minq(gamma,c,G,xu,xo,prt,xx);
% minimizes an affine quadratic form subject to simple bounds,
% using coordinate searches and reduced subspace minimizations
%    min    fct = gamma + c^T x + 0.5 x^T G x 
%    s.t.   x in [xu,xo]    % xu<=xo is assumed
% where G is a symmetric n x n matrix, not necessarily definite
% (if G is indefinite, only a local minimum is found)
%
% if G is sparse, it is assumed that the ordering is such that
% a sparse modified Cholesky factorization is feasible
%
% prt	printlevel
% xx	guess (optional)
%
% x	minimizer (but unbounded direction if ier=1)
% fct	optimal function value
% ier	0  (local minimizer found)
% 	1  (unbounded below)
% 	99 (maxit exceeded)
% 	-1 (input error)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% minqdef.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [x,y,ier]=minqdef(c,G,A,b,eq,prt,xx);
% minimizes a definite quadratic form subject to linear constraints
%    min    fct = c^T x + 0.5 x^T G x 
%    s.t.   A x >= b, with equality at indices with eq=1
% where G is a definite symmetric n x n matrix
%
% if A is sparse, it is assumed that the ordering is such that the
% sparse Cholesky factorizations of G and AG^(-1)A^T are feasible
%
% eq    characteristic vector of equalities
% prt	printlevel
% xx	guess (optional)
%
% x	minimizer (but unbounded direction if ier=1)
% y     Lagrange multiplier satisfying the KKT conditions
%       Gx=A^Ty-c, inf(y,Ax-b)=0 at indices with eq=0
% ier	0  (global minimizer found)
% 	1  (approximate solution; feasible set probably empty)
% 	99 (approximate solution; maxit exceeded)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% minqsep.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [x,y,ier]=minqsep(c,d,A,b,eq,prt,xx);
% minimizes a definite separable quadratic form 
% subject to linear constraints
%    min    fct= c^T x + 0.5 x^T D x 
%    s.t.   A*x>=b, with equality at indices with eq=1
% where D=diag(d) is a definite n x n diagonal matrix
%
% if A is sparse, it is assumed that the ordering is such that
% a sparse Cholesky factorization of AA^T is feasible
%
% eq    characteristic vector of equalities
% prt	printlevel
% xx	guess (optional)
%
% x	minimizer (but unbounded direction if ier=1)
% y     Lagrange multiplier satisfying the KKT conditions
%       Dx=A^Ty-c, inf(y,Ax-b)=0 at indices with ind=0
% ier	0  (global minimizer found)
% 	1  (approximate solution; feasible set probably empty)
% 	99 (approximate solution; maxit exceeded)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% rls.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [x,act]=rls(A,b,fac);
% function [x,act]=rls(A,b,x0,r);
% minimizes ||Ax-b||_2^2  s.t. |x-x0|<=r
% act=(is some constraint active?)
%
% if nargin<4, x0=0 and r(k)=fac*||b||/||A(:,k)||
% (default: fac=1000);
%
