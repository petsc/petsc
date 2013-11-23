

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% minqsep.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [x,y,ier]=minqsep(c,d,A,b,eq,prt,xx);
% minimizes a definite separable quadratic form
% subject to linear constraints
%    min    fct = c^T x + 0.5 x^T G x 
%    s.t.   A x >= b, with equality at indices with eq=1
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
%       Dx=A^Ty-c, inf(y,Ax-b)=0 at indices with eq=0
% ier	0  (global minimizer found)
% 	1  (approximate solution; feasible set probably empty)
% 	99 (approximate solution; maxit exceeded)
%
% Method: apply minq.m to the dual
%    min  0.5*(A^Ty-c)^TD^(-1)(A^Ty-c)-b^Ty 
%    s.t. y(~eq)>=0
% x is recovered as x=D^(-1)(A^Ty-c)
%
function [x,y,ier]=minqsep(c,d,A,b,eq,prt,xx);


if min(d)<=0, error('diagonal must be positive'); end;

[m,n]=size(A);
D=spdiags(d,0,n,n);
G=A*(D\A');
cc=-b-A*(c./d);
yo=inf+zeros(m,1);
yu=zeros(m,1);yu(eq)=-yo(eq);

[y,fct,ier]=minq(0,cc,G,yu,yo,prt);
x=(A'*y-c)./d;
if ier==99, return; end;

% check for accuracy
res=A*x-b;ressmall=nnz(A)*eps*(abs(A)*abs(x)+abs(b));
res(~eq)=min(res(~eq),0);
if prt, 
  disp('residual (first row) small if comparable to second row')
  disp([res,ressmall]')
end;
if min(abs(res)<=ressmall),
  % accuracy satisfactory
  ier=0;
  return;
end;

% one step of iterative refinement
if prt, 
  disp('one step of iterative refinement')
end;
[dy,fct,ier]=minq(0,-res,G,yu-y,yo-y,prt);
x=x+(A'*dy)./d;
y=y+dy;

% check for accuracy
res=A*x-b;ressmall=nnz(A)*eps*(abs(A)*abs(x)+abs(b));
res(~eq)=min(res(~eq),0);
if min(abs(res)<=sqrt(nnz(A))*ressmall),
  % accuracy satisfactory
  ier=0;
else
  % feasible set probably empty
  ier=1;
end;

