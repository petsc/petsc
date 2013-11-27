

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ldlup.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [L,d,p]=ldlup(L,d,j,g)
% updates LDL^T factorization when a unit j-th row and column
% are replaced by column g 
% if the new matrix is definite (signalled by p=[]);
% otherwise, the original L,d and 
% a direction p of null or negative curvature are returned
%
% d contains diag(D) and is assumed positive
% Note that g must have zeros in other unit rows!!!
%
function [L,d,p]=ldlup(L,d,j,g);

p=[];

test=0;
if test, 
  disp('enter ldlup')
  A=L*diag(d)*L';A(:,j)=g;A(j,:)=g'; 
end;

n=size(d,1);
I=1:j-1;K=j+1:n;
if j==1,
  v=zeros(0,1);
  del=g(j);
  if del<=n*eps, 
    p=[1;zeros(n-1,1)]; 
    if test, 
      A,p
      Nenner=abs(p)'*abs(A)*abs(p);
      if Nenner==0, indef1=0 ,else indef1=(p'*A*p)/Nenner, end;
      disp('leave ldlup at 1')
    end;
    return; 
  end;
  w=g(K)/del;
  L(j,I)=v';
  d(j)=del;
  if test, 
    A1=L*diag(d)*L',A 
    quot=norm(A1-A,1)/norm(A,1), 
    disp('leave ldlup at 3')
  end;
  return;  
end;

% now j>1, K nonempty
LII=L(I,I);
u=LII\g(I);
v=u./d(I);
del=g(j)-u'*v;
if del<=n*eps,
  p=[LII'\v;-1;zeros(n-j,1)];
  if test, 
    A,p
    indef1=(p'*A*p)/(abs(p)'*abs(A)*abs(p))
    disp('leave ldlup at 2')
  end;
  return;
end;
LKI=L(K,I);
w=(g(K)-LKI*u)/del;
[LKK,d(K),q]=ldlrk1(L(K,K),d(K),-del,w);
if isempty(q),
  % work around expensive sparse L(K,K)=LKK
  L=[L(I,:);
     v', 1,L(j,K);
     LKI,w,LKK];
  d(j)=del;
  if test, 
    A1=L*diag(d)*L',A 
    quot=norm(A1-A,1)/norm(A,1), 
    disp('leave ldlup at 4')
  end;
else
  % work around expensive sparse L(K,K)=LKK
  L=[L(1:j,:);
     LKI,L(K,j),LKK];
  pi=w'*q;
  p=[LII'\(pi*v-LKI'*q);-pi;q];
  if test, 
    indef2=(p'*A*p)/(abs(p)'*abs(A)*abs(p)), 
    disp('leave ldlup at 5')
  end;
end;

