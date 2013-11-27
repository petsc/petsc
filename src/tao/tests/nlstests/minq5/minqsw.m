

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% minq.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [x,fct,ier,nsub]=minq(gam,c,G,xu,xo,prt,xx);
% minimizes an affine quadratic form subject to simple bounds,
% using coordinate searches and reduced subspace minimizations
% using LDL^T factorization updates
%    min    fct = gam + c^T x + 0.5 x^T G x 
%    s.t.   x in [xu,xo]    % xu<=xo is assumed
% where G is a symmetric n x n matrix, not necessarily definite
% (if G is indefinite, only a local minimum is found)
%
% if G is sparse, it is assumed that the ordering is such that
% a sparse modified Cholesky factorization is feasible
%
% prt	printlevel
% xx	initial guess (optional)
%
% x	minimizer (but unbounded direction if ier=1)
% fct	optimal function value
% ier	0  (local minimizer found)
% 	1  (unbounded below)
% 	99 (maxit exceeded)
% 	-1 (input error)
%
% calls getalp.m, ldl*.m, minqsub.m, pr01.m
%
function [x,fct,ier,nsub]=minqsw(gam,c,G,xu,xo,prt,xx);

% This is minq with changes to:
% 1) max number of iterations,
% 2) display option for exceeding maxit
% Search for 'SW' to find specific lines


%%% initialization %%%
%disp('####################################################');
%disp('######################  minq  ######################');
%disp('####################################################');
convex=0;
n=size(G,1);

if prt>0, 
  % show printlevel
  printlevel=prt

  % check input data for consistency
  ier=0;
  if size(G,2)~=n, 
    ier=-1;disp('minq: Hessian has wrong dimension');
    x=NaN+zeros(n,1);fct=NaN;nsub=-1;
    return;
  end;
  if size(c,1)~=n | size(c,2)~=1, 
    ier=-1;disp('minq: linear term has wrong dimension');
  end;
  if size(xu,1)~=n | size(xu,2)~=1, 
    ier=-1;disp('minq: lower bound has wrong dimension');
  end;
  if size(xo,1)~=n | size(xo,2)~=1, 
    ier=-1;disp('minq: lower bound has wrong dimension');
  end;
  if exist('xx')==1,
    if size(xx,1)~=n | size(xx,2)~=1, 
      ier=-1;disp('minq: lower bound has wrong dimension');
    end;
  end;
  if ier==-1,
    x=NaN+zeros(n,1);fct=NaN;nsub=-1;
    return;
  end;
end;

maxit=3*n;       	% maximal number of iterations
maxit=5*n;       	% maximal number of iterations % Changed by SW
	% this limits the work to about 1+4*maxit/n matrix multiplies
	% usually at most 2*n iterations are needed for convergence
nitrefmax=3;		% maximal number of iterative refinement steps

% initialize trial point xx, function value fct and gradient g

if nargin<7,
  % cold start with absolutely smallest feasible point
  xx=zeros(n,1);
end;
% force starting point into the box
xx=max(xu,min(xx,xo));

% regularization for low rank problems
hpeps=100*eps;	% perturbation in last two digits
G=G+spdiags(hpeps*diag(G),0,n,n);

% initialize LDL^T factorization of G_KK
K=logical(zeros(n,1));	% initially no rows in factorization
if issparse(G), L=speye(n); else L=eye(n); end;
dd=ones(n,1);	

% dummy initialization of indicator of free variables
% will become correct after first coordinate search
free=logical(zeros(n,1)); 
nfree=0;
nfree_old=-1;

fct=inf; 		% best function value
nsub=0;			% number of subspace steps
unfix=1;		% allow variables to be freed in csearch?
nitref=0;		% no iterative refinement steps so far
improvement=1;		% improvement expected

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop: alternating coordinate and subspace searches 
while 1,
  if prt>1, disp('enter main loop'); end;
  if norm(xx,inf)==inf, error('infinite xx in minq.m'); end;
  g=G*xx+c;
  fctnew=gam+0.5*xx'*(c+g);
  if ~improvement,
    % good termination 
    if prt, 
      disp('terminate: no improvement in coordinate search'); 
    end;
    ier=0; break; 
  elseif nitref>nitrefmax,
    % good termination 
    if prt, disp('terminate: nitref>nitrefmax'); end;
    ier=0; break; 
  elseif nitref>0 & nfree_old==nfree & fctnew >= fct,
    % good termination 
    if prt, 
      disp('terminate: nitref>0 & nfree_old==nfree & fctnew>=fct'); 
    end;
    ier=0; break; 
  elseif nitref==0,
    x=xx;
    fct=min(fct,fctnew);
    if prt>1, fct, end;
    if prt>2, X=x', fct, end;
  else % more accurate g and hence f if nitref>0
    x=xx;
    fct=fctnew;
    if prt>1, fct, end;
    if prt>2, X=x', fct, end;    
  end;
  if nitref==0 & nsub>=maxit, 
    if prt,
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'); 
      disp('!!!!!           minq          !!!!!'); 
      disp('!!!!! incomplete minimization !!!!!'); 
      disp('!!!!!   too many iterations   !!!!!'); 
      disp('!!!!!     increase maxit      !!!!!'); 
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');
    else
        % Changed by SW
        % disp('iteration limit exceeded');
    end;
    ier=99;
    break;
  end;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % coordinate search
  count=0; 	% number of consecutive free steps
  k=0;     	% current coordinate searched
  while 1,
    while count<=n,
      % find next free index (or next index if unfix)
      count=count+1;
      if k==n, k=0; end;
      k=k+1;
      if free(k) | unfix, break; end;
    end;
    if count>n, 
      % complete sweep performed without fixing a new active bound
      break; 
    end;
    q=G(:,k);
    alpu=xu(k)-x(k); alpo=xo(k)-x(k); % bounds on step

    % find step size
    [alp,lba,uba,ier]=getalp(alpu,alpo,g(k),q(k));
    if ier,
      x=zeros(n,1); 
      if lba, x(k)=-1; else x(k)=1; end;
      if prt, 
        gTp=g(k),pTGp=q(k),quot=pTGp/norm(G(:),inf)
        disp('minq: function unbounded below in coordinate direction'); 
        disp('      unbounded direction returned'); 
        disp('      possibly caused by roundoff'); 
      end;
      if prt>1, 
        disp('f(alp*x)=gam+gam1*alp+gam2*alp^2/2, where'); 
        gam1=c'*x
        gam2=x'*(G*x)
        ddd=diag(G);
        min_diag_G=min(ddd)
        max_diag_G=max(ddd)
      end;
      return;
    end;
    xnew=x(k)+alp;
    if prt & nitref>0,
      xnew,alp
    end;
    
    if lba | xnew<=xu(k),
      % lower bound active
      if prt>2, disp([num2str(k), ' at lower bound']); end;
      if alpu~=0,
        x(k)=xu(k);
        g=g+alpu*q;
        count=0;
      end;
      free(k)=0;
    elseif uba | xnew>=xo(k),
      % upper bound active
      if prt>2, disp([num2str(k), ' at upper bound']); end;
      if alpo~=0,
        x(k)=xo(k);
        g=g+alpo*q;
        count=0;
      end;
      free(k)=0;
    else
      % no bound active
      if prt>2, disp([num2str(k), ' free']); end;
      if alp~=0,
        if prt>1 & ~free(k), 
          unfixstep=[x(k),alp], 
        end;
        x(k)=xnew;
        g=g+alp*q;
        free(k)=1;
      end;
    end;

  end;
  % end of coordinate search

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  nfree=sum(free);
  if (unfix & nfree_old==nfree),
    % in exact arithmetic, we are already optimal
    % recompute gradient for iterative refinement
    g=G*x+c;
    nitref=nitref+1;
    if prt>0,
      disp('optimum found; iterative refinement tried');
    end;
  else
    nitref=0;
  end;
  nfree_old=nfree;
  gain_cs=fct-gam-0.5*x'*(c+g);
  improvement=(gain_cs>0 | ~unfix);

  if prt, 
    % print (0,1) profile of free and return the number of nonzeros
    nfree=pr01('csrch ',free);
  end; 
  if prt, gain_cs, end;
  if prt>2, X=x', end;

  % subspace search
  xx=x; 
  if ~improvement | nitref>nitrefmax,
    % optimal point found - nothing done
  elseif nitref>nitrefmax,
    % enough refinement steps - nothing done
  elseif nfree==0,
    % no free variables - no subspace step taken
    if prt>0,
      disp('no free variables - no subspace step taken')
    end;
    unfix=1;
  else
    % take a subspace step
    minqsub; 
    if ier, return; end;
  end;

  if prt>0, 
    % print (0,1) profile of free and return the number of nonzeros
    nfree=pr01('ssrch ',free);
    disp(' ');
    if unfix & sum(nfree)<n,
      disp('bounds may be freed in next csearch'); 
    end;
  end; 


end;
% end of main loop
if prt>0, 
  fct
  disp('################## end of minq ###################');
end;
