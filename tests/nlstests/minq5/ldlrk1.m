

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ldlrk1.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [L,d,p]=ldlrk1(L,d,alp,u)
% computes LDL^T factorization for LDL^T+alp*uu^T
% if alp>=0 or if the new factorization is definite 
% (both signalled by p=[]);
% otherwise, the original L,d and 
% a direction p of null or negative curvature are returned
%
% d contains diag(D) and is assumed positive
% 
% does not work for dimension 0
%
function [L,d,p]=ldlrk1(L,d,alp,u);

test=0; % only for testing the routine
if test, 
  disp('enter ldlrk1')
  A=L*diag(d)*L'+(alp*u)*u';
end;

p=[];
if alp==0, return; end;

n=size(u,1);
neps=n*eps;

% save old factorization
L0=L;d0=d;

% update
for k=find(u~=0)',
  del=d(k)+alp*u(k)^2;
  if alp<0 & del<=neps,
    % update not definite
    p=zeros(n,1);p(k)=1;
    p(1:k)=L(1:k,1:k)'\p(1:k);
    % restore original factorization
    L=L0;d=d0;
    if test, 
      indef=(p'*(A*p))/(abs(p)'*(abs(A)*abs(p))) 
      disp('leave ldlrk1 at 1') 
    end;
    return;
  end;
  q=d(k)/del;
  d(k)=del;
  % in C, the following 3 lines would be done in a single loop
  ind=k+1:n;
  c=L(ind,k)*u(k);
  L(ind,k)=L(ind,k)*q+(alp*u(k)/del)*u(ind,1);
  u(ind,1)=u(ind,1)-c;
  alp=alp*q;
  if alp==0, break; end;
end;
if test, 
  A1=L*diag(d)*L',A 
  quot=norm(A1-A,1)/norm(A,1)
  disp('leave ldlrk1 at 2')
end;

