% MNHnls.m                             Created by Stefan Wild & Jorge More'
% Version 0.1,    Modified 9/24/08
% --INPUTS-----------------------------------------------------------------
% func    [f h] Function handle so that func(x) evaluates f (@calfun)
% X0      [dbl] [1-by-n] Initial point  (zeros(1,n))
% n       [int] Dimension (number of continuous variables)
% npmax   [int] Maximum number of interpolation points (>n+1) (2*n+1)
% nfmax   [int] Maximum number of function evaluations (>n+1) (100)
% delta   [dbl] Positive trust region radius (.1)
% gradtol [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
% m       [int] Number of components
%
% --OUTPUTS----------------------------------------------------------------
% X       [dbl] [nfmax-by-n] Stores the evaluation point locations
% F       [dbl] [nfmax-by-1] Stores the function values of evaluated points
% flag    [dbl] Termination criteria flag:
%               = 0 normal termination because of grad,
%               > 0 exceeded nfmax evals,   flag = norm of grad at final xb
%               < 0 delta got "too small", -flag = norm of grad at final xb
%
function [X,F,flag]=MNHnls(func,X0,n,npmax,nfmax,gradtol,delta,m,mtype)
% --DEPENDS ON-------------------------------------------------------------
% mgqt     : Subproblem solver
% AffPoints, MorePoints : Updates interpolation set
% mfQuad  : Forms and fits quadratic model
%
% --INTERMEDIATE-----------------------------------------------------------
% D       [dbl] [np-by-n] Stores the evaluation point displacements
% G       [dbl] [n-by-1]  Model gradient at Xk
% GPoints [dbl] [n-by-n]  Displacements for geometry improving points
% H       [dbl] [n-by-n]  Model Hessian at Xk
% Modeld  [dbl] [1-by-n]  Unit direction to improve model
% ModelIn [int] [npmax-by-1] Integer vector of model interpolation indices
% Xsp     [dbl] [1-by-n] subproblem solution
% c       [dbl] Model value at X(xkin,:)
% mdec    [dbl] Decrease predicted by the model
% nf      [int] Counter for the number of function evaluations
% ng      [dbl] =norm(G)
% nm      [int] Number of model points evaluated
% np      [int] Number of model interpolation points
% rho     [dbl] Ratio of actual decrease to model decrease
% xkin    [int] Index of current center
% valid   [log] Flag saying if model is valid within c1*delta
%
% --INTERNAL PARAMETERS----------------------------------------------------
maxdelta = (1e3)*delta; % [dbl] Maximum trust region radius (>=delta) (1e3)
mindelta = (1e-13)*delta; % [dbl] Minimum tr radius (technically 0) (1e-6)
%beta = .001;    % [dbl] Parameter for the model gradient/delta (.001)
c1 = sqrt(n);     % [dbl] Factor for checking validity (3)
c2 = 100;       % [dbl] Factor for linear poisedness (3)
theta1 = 1e-5;  % [dbl] Pivot threshold for validity   (1e-5)
theta2 = .0001;  % [dbl] Pivot threshold for additional points (.001)
gam0 = .5;      % [dbl] Parameter for shrinking delta (<1)  (.5)
gam1 = 2;       % [dbl] Parameter for enlarging delta (>1)  (2)
eta0 = 0;       % [dbl] Parameter 1 for accepting point (0<=eta0<eta1) (0)
eta1 = .1;      % [dbl] Parameter 2 for accepting point (eta0<eta1<1) (.2)
rtol = 0.001;   % [dbl] Parameter used by gqt (0.001)
itmax = 50;     % [int] Parameter used by gqt (50)
% -------------------------------------------------------------------------

%%% START OF SETUP PHASE: Initialize and evaluate the first n+1 points %%%%
X = zeros(nfmax,n);     % Stores the evaluation point locations
F = zeros(nfmax,m);     % Stores the function values of evaluated points
Fres = zeros(nfmax,1);     % Stores the residual values of evaluated points
nm = 0; % Counter for the number of model points evaluated

X(1,:) = X0;
X(2:n+1,:) = repmat(X(1,:),n,1)+delta*eye(n);
for nf = 1:n+1
    F(nf,:) = func(X(nf,:))';
    Fres(nf) = sum(F(nf,:).^2);
end

[c,xkin] = min(Fres(1:n+1));    % Find index of best f in setup phase
ModelIn = [1:xkin-1, xkin+1:nf]'; % Vector of Model Indices (not Xk)
D = (X(1:nf,:)-repmat(X(xkin,:),nf,1))/delta; % Matrix of displacements from Xk

% Determine the initial quadratic models:
%C = F(xkin,:);
G = D(ModelIn,:) \ (F(ModelIn,1:m)-repmat(F(xkin,1:m),n,1));  
H = zeros(n,n,m);

if mtype==1
    Cres = Fres(xkin);
    Gres = G*F(xkin,1:m)';
    Hres = zeros(n);
    for i=1:m
        Hres = Hres + F(xkin,i)*H(:,:,i);
    end
    Hres = Hres + G*G';
else
    Cres = Fres(xkin);
    Gres = D(ModelIn,:) \ (Fres(ModelIn)-Fres(xkin));
    Hres = zeros(n);
end

valid = true;                      % First model is valid by construction
ng = norm(Gres)*delta;

% Output stuff: -------------------------------------------------------
np = n+1; %for first output
disp('  nf    delta     fl  np       f0           g0       ierror');
progstr = '%4i %12.4e %2i %3i  %11.5e %12.4e %11.3e';  % For line-by-line output.
disp(sprintf(progstr, nf, delta,valid, np, Fres(1), ng,0)); % screen
%----------------------------------------------------------------------
%%% END OF SETUP PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while nf<nfmax
    % 1. Final Criticality Test
    if ng<gradtol  %** Still need to change: valid in a smaller region %!
        delta2 = gradtol;
        [ModelIn,Modeld] = AffPoints(X(1:nf,:),eye(n),[],[],xkin,delta2,theta1,c1);
        if size(ModelIn,1)==n % If valid, return
            disp('g is sufficiently small')
            X=X(1:nf,:); F=F(1:nf,:); flag = 0;
            return;
        else        % Make more valid
            modelval = zeros(size(Modeld,1),1);
            for i = 1: size(Modeld,1)
                if Modeld(i,:)*Gres>0 % Model says use the other direction!
                    Modeld(i,:) = -Modeld(i,:);
                end
                modelval(i) = c+Modeld(i,:)*(Gres+.5*Hres*Modeld(i,:)'); %% careful this should have a ratio of the two deltas
            end
            [a,b]=min(modelval);
            Modeldir = Modeld(b,:);

            nf = nf + 1;
            nm = nm+1; % Use to count model numbers for now
            X(nf,:) = X(xkin,:) + delta2*Modeldir;
            F(nf,:) = func(X(nf,:))';
            Fres(nf) = sum(F(nf,:).^2);
            %rho
            disp(sprintf('***** %i value of Critical Point: %g',nm,Fres(nf)));
            %disp(sprintf('***** %i value of SP So Point: %g',nm,F(nf-1)));
        end
    end



    % 2. Solve the subproblem min{Q(s): ||s|| <= delta}
        [Xsp,mdec,dum1,dum2] = gqt(Hres,Gres,1,rtol,itmax,ng);
        mdec = -mdec; % Correct the sign

    % 3a. Evaluate the function at the new point
    nf = nf + 1;    X(nf,:) = X(xkin,:) + delta*Xsp';   
    F(nf,:) = func(X(nf,:))';
    Fres(nf) = sum(F(nf,:).^2);
    rho = (Fres(xkin) - Fres(nf))/mdec;
    %%cmcdata_save

    % Output stuff: -------------------------------------------------------
    for i=1:size(ModelIn,1)
        ierror(i,1) = Cres+(X(ModelIn(i),:)-X(xkin,:))*(Gres+.5*Hres*(X(ModelIn(i),:)-X(xkin,:))'/delta)/delta-Fres(ModelIn(i));
    end
    %disp('  nf   delta   valid  np f0           g0         ierror');
    ierror = norm(ierror./max(abs(F(ModelIn)),10^-26),'inf'); % Interpolation error.
    disp(sprintf(progstr, nf, delta, valid, np, Fres(xkin), ng, ierror)); % screen
    %----------------------------------------------------------------------

    % 3b. Update the center
    if (rho >= eta1)  || ((rho>eta0) && (valid))
        % 5a. Update model to reflect new base point
        Displace = (X(nf,:)-X(xkin,:))/delta; %%
        Cres = Cres+Displace*Gres+.5*Displace*Hres*Displace'; % Need this for ierror
        Gres = Gres + Hres*Displace'; % Need to do this since we use Gres below
        xkin = nf; % Change current center
    end

    % 4. Evaluate at a model-improving point if necessary
    if ~(valid)
        [ModelIn,Modeld] = AffPoints(X(1:nf,:),eye(n),[],[],xkin,delta,theta1,c1);
        if size(ModelIn,1)<n    % Need to check because model may now be valid
            modelval = zeros(size(Modeld,1),1);
            for i = 1: size(Modeld,1)
                if Modeld(i,:)*Gres>0 % Model says use the other direction!
                    Modeld(i,:) = -Modeld(i,:);
                end
                modelval(i) = Cres+Modeld(i,:)*(Gres+.5*Hres*Modeld(i,:)'); %% %! really need to add cres here?!?
            end
            [a,b]=min(modelval);
            Modeldir = Modeld(b,:);

            nf = nf + 1;
            nm = nm+1; % Use to count model numbers for now
            X(nf,:) = X(xkin,:) + delta*Modeldir;
            F(nf,:) = func(X(nf,:))';
            Fres(nf) = sum(F(nf,:).^2);
            %rho
            %disp(sprintf('***** %i value of Model Point: %g',nm,F(nf)));
            %disp(sprintf('***** %i value of SP So Point: %g',nm,F(nf-1)));
        end
    end

    % 5. Update the trust region radius:
    if (rho >= eta1)  &&   norm(Xsp)>.5*delta
        delta = min(delta*gam1,maxdelta);
    elseif (valid)
        delta = max(delta*gam0,mindelta);
    end

    % 6a. Compute the next interpolation set.
    [ModelIn,Modeld,Q,R] = AffPoints(X(1:nf,:),eye(n),[],[],xkin,delta,theta1,c1);
    if size(ModelIn,1)==n
        valid = true;
    else
        valid = false;
        [ModelIn,GPoints] = AffPoints(X(1:nf,:),Q,R,ModelIn,xkin,delta,theta1,c2);
        np = size(ModelIn,1);
        for i=1:n-np
            if GPoints(i,:)*Gres>0 % Model says use the other direction!
                GPoints(i,:) = -GPoints(i,:);
            end
            nf = nf + 1;
            ModelIn(np+i,1) = nf;
            X(nf,:) = X(xkin,:) + delta*GPoints(i,:);
            F(nf,:) = func(X(nf,:))';
            Fres(nf) = sum(F(nf,:).^2);
            disp(sprintf('***** value of Geometry Point: %g',Fres(nf)));
            %**eventually may need to check if we're valid here
        end
    end
    ModelIn = [xkin; ModelIn];  % Add center point to model indices

    [ModelIn,L,Z,M,N] = MorePoints(X(1:nf,:),ModelIn,xkin,npmax,delta,theta2,c2);

    % 6b. Update the quadratic model
    [dum1,Gres,Hres,G,H] = getQuadnls(L,Z,M,N,F(ModelIn,:),Fres(ModelIn),F(xkin,:),mtype,m);
    
    
    if 1==0 

    np = size(ModelIn,1); % Number of model interpolation points
    % Test interp error:
    ERROR =zeros(np,m);
    C=F(xkin,:);%+Cdel; % Don't need this???
    for i=1:np
        D(i,:) = (X(ModelIn(i),:)-X(xkin,:))/delta;
        for j=1:m
            ERROR(i,j) = C(j)+D(i,:)*(G(:,j)+.5*H(:,:,j)*D(i,:)')-F(ModelIn(i),j);
        end
    end

    norm(ERROR./((F(ModelIn,:).*F(ModelIn,:)~=0)+(F(ModelIn,:)==0)));

    end
    %[dum1,G,H,C] = mfQuad(F(ModelIn),D',delta);  %** Why does this dep on Delta?
    Cres = Fres(xkin);
    ng = norm(Gres)*delta;
end
disp('Number of function evals exceeded');
flag = ng;
