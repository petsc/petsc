

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% rls_test.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test rls.m
% is a factor 20-30 slower than A\b but much more stable


n=400;
m=n+5;
sig=1e-7;
A=rand(m,n);
%A(:,1)=A(:,n-4)+A(:,2)+sig*randn(m,1); % produce rank deficiency
A(:,3)=A(:,n-5)+A(:,n-2)+sig*randn(m,1); % produce rank deficiency
A(6,:)=A(7,:)+A(5,:)+sig*randn(1,n); % produce rank deficiency
x0=randn(n,1);
b=A*x0+1e-2*randn(m,1);
% b=randn(m,1);
tic;x=A\b;res1=norm(b-A*x);,tim1=toc
tic;[xx,act]=rls(A,b);res2=norm(b-A*xx),tim2=toc
disp('---')
[x0,x xx]'
res1,res2,act,
growth_ratio=norm(x)/norm(xx)
slow=tim2/tim1
return




% test for a suitable value of gam
% gam=1000 is fully adequate
figure(1);
ntim=40;
gam=zeros(ntim,ntim);
nlist=[10,10,10,10,10,20,20,20,20,20,50,50,50,50,50,100,100,100,100,100,200,200,200,200,200];
for tim0=1:length(nlist),
  n=nlist(tim0);
  for tim1=1:ntim,tim1
    A=rand(n);
    a=sqrt(diag(A'*A));
    for tim2=1:ntim,
      b=rand(n,1);
      x=A\b;
      gam(tim1,tim2)=max(a.*abs(x))/norm(b);
    end;
  end;
  str='ooooo+++++xxxxx.....-----';
  t=tim1*tim2;
  perc=[1:t]*(100/t);
  semilogy(perc,sort(gam(:)),str(tim0));hold on
  set(gca,'xlim',[95,99])
  drawnow
  %input('next>');
end;

