

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% minqdef.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [x,y,ier]=minqdef(c,G,A,b,eq,prt,xx);
% minimizes a definite quadratic form subject to linear constraints
%    min    fct = c^T x + 0.5 x^T G x 
%    s.t.   A x >= b, with equality at indices with eq=1
% where G is a definite symmetric n x n matrix
%
% if A is sparse, it is assumed that the ordering is such that
% sparse Cholesky factorization of G and AG^(-1)A^T are feasible
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
% Method: apply minq.m to the dual
%    min  0.5*(A^Ty-c)^TG^(-1)(A^Ty-c)-b^Ty 
%    s.t. y(~eq)>=0
% x is recovered as x=G^(-1)(A^Ty-c)
%
function [x,y,ier]=minqdef(c,G,A,b,eq,prt,xx);


R=chol(G);

[m,n]=size(A);
A0=A/R;
GG=A0*A0';
c0=R'\c;
cc=-b-A0*c0;
yo=inf+zeros(m,1);
yu=zeros(m,1);yu(eq)=-yo(eq);

[y,fct,ier]=minq(0,cc,GG,yu,yo,prt);
x=R\(A0'*y-c0);
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
[dy,fct,ier]=minq(0,-res,GG,yu-y,yo-y,prt);
x=x+R\(A0'*dy);
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

