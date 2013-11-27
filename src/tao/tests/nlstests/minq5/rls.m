

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
function [x,act]=rls(A,b,x0,r);

% set defaults
if     nargin<3, fac=1000;x0=zeros(size(A,2),1); 
elseif nargin<4, fac=x0;x0=zeros(size(A,2),1);
else             b=b-A*x0;
end;

x=x0;
aa=diag(A'*A);

% handle zero columns directly
ind=find(aa>0);
n=length(ind);
if n==0, act=1; return; end;

A=A(:,ind);
gamma=b'*b;
if gamma==0, act=1; return; end;
if nargin<4, r=fac*sqrt(gamma./aa(ind));
else         r=r(ind);
end;

% now we need to minimize ||Az-b||^2  s.t. |z|<=r
prt=0;
[z,fct,ier]=minq(0,-A'*b,A'*A,-r,r,prt);
act=max(abs(z)./r);
x(ind)=x(ind)+z;



