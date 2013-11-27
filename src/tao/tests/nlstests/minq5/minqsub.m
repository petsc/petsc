

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% minqsub.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% patch for minq.m containing the subspace search
%

nsub=nsub+1;

if prt>0, 
  fct_cs=gam+0.5*x'*(c+g);
  format='*** nsub = %4.0f fct = %15.6e fct_cs = %15.6e';
  disp(sprintf(format,nsub,fct,fct_cs)); 
end;

% downdate factorization
for j=find(free<K)', 	% list of newly active indices
  [L,dd]=ldldown(L,dd,j);
  K(j)=0;
  if prt>10,
    disp('downdate');
    fact_ind=find(K)' 
  end;
end;

% update factorization or find indefinite search direchtion
definite=1;
for j=find(free>K)', 	% list of newly freed indices
  % later: speed up the following by passing K to ldlup.m!
  p=zeros(n,1);
  if n>1, p(K)=G(find(K),j); end;
  p(j)=G(j,j);
  [L,dd,p]=ldlup(L,dd,j,p);
  definite=isempty(p);
  if ~definite, 
    if prt,disp('indefinite or illconditioned step');end; 
    break; 
  end;
  K(j)=1;
  if prt>10,
    disp('update');
    fact_ind=find(K)' 
  end;
end;

if definite,
  % find reduced Newton direction 
  p=zeros(n,1);
  p(K)=g(K);
  p=-L'\((L\p)./dd);
  if prt>10, 
    disp('reduced Newton step');
    fact_ind=find(K)' 
  end;
end;

if prt>2, input('continue with return>'); end;

% set tiny entries to zero 
p=(x+p)-x;
ind=find(p~=0);
if isempty(ind); 
  % zero direction
  if prt,disp('zero direction');end;
  unfix=1;
  return;
end;

% find range of step sizes 
pp=p(ind);
oo=(xo(ind)-x(ind))./pp;
uu=(xu(ind)-x(ind))./pp;
alpu=max([oo(pp<0);uu(pp>0);-inf]);
alpo=min([oo(pp>0);uu(pp<0);inf]);
if alpo<=0 | alpu>=0, 
  error('programming error: no alp'); 
end;

% find step size 
gTp=g'*p;
agTp=abs(g)'*abs(p);
if abs(gTp)<100*eps*agTp,
  % linear term consists of roundoff only
  gTp=0;
end;
pTGp=p'*(G*p);
if convex, pTGp=max(0,pTGp); end;
if ~definite & pTGp>0, 
  if prt, disp(['tiny pTGp = ',num2str(pTGp),' set to zero']); end;
  pTGp=0; 
end;
[alp,lba,uba,ier]=getalp(alpu,alpo,gTp,pTGp);
if ier,
  x=zeros(n,1); 
  if lba, x=-p; else x=p; end;
  if prt, 
    qg=gTp/agTp
    qG=pTGp/(norm(p,1)^2*norm(G(:),inf))
    lam=eig(G);
    lam1=min(lam)/max(abs(lam))
    disp('minq: function unbounded below'); 
    disp('  unbounded subspace direction returned')
    disp('  possibly caused by roundoff'); 
    disp('  regularize G to avoid this!'); 
  end;
  if prt>1,
    disp('f(alp*x)=gam+gam1*alp+gam2*alp^2/2, where'); 
    gam1=c'*x
    rel1=gam1/(abs(c)'*abs(x))
    gam2=x'*(G*x)
    if convex, gam2=max(0,gam2); end;
    rel2=gam2/(abs(x)'*(abs(G)*abs(x))) 
    ddd=diag(G);
    min_diag_G=min(ddd)
    max_diag_G=max(ddd)
  end;
  return;
end;
unfix=~(lba | uba);  % allow variables to be freed in csearch?

% update of xx
for k=1:length(ind),
  % avoid roundoff for active bounds 
  ik=ind(k);
  if alp==uu(k),
    xx(ik)=xu(ik);
    free(ik)=0;
  elseif alp==oo(k),
    xx(ik)=xo(ik); 
    free(ik)=0;
  else
    xx(ik)=xx(ik)+alp*p(ik);
  end;
  if abs(xx(ik))==inf, 
    ik,alp,p(ik),
    error('infinite xx in minq.m'); 
  end;

end;
nfree=sum(free);
subdone=1;

