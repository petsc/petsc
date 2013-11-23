% MorePoints.m                         Created by Stefan Wild & Jorge More'
% Version 0.0,    Modified 1/10/08
% This routine finds additional indices ModelIn used by MNH.
%
% INPUTS-------------------------------------------------------------------
% X       [dbl] [nf-by-n] Stores the evaluation point locations
% xkin    [int] Index of current center
% npmax   [int] Maximum number of interpolation points (>n+1)
% delta   [dbl] Positive trust region radius
% theta2  [dbl] Pivot threshold for additional points
% c2      [dbl] Factor for checking validity
%
% OUTPUTS------------------------------------------------------------------
% ModelIn [int] [np-by-1] Integer vector of model interpolation indices
%
function [ModelIn,L,Z,M,N] = MorePoints(X,ModelIn,xkin,npmax,delta,theta2,c2)
% DEPENDS ON---------------------------------------------------------------
% phi2eval   : Computes standard quadratic basis vector
%
% --INTERMEDIATE-----------------------------------------------------------
% Q       [dbl] [mm-by-mm]  %** look into
% R       [dbl] [mm-by-n+1] %** look into
% mm      [int] Number of components in the quadratic basis, .5*(n+1)*(n+2)
% n       [int] Dimension (number of continuous variables)
% nf      [int] Counter for the number of function evaluations
% np      [int] Number of model interpolation points
% -------------------------------------------------------------------------

[nf,n] = size(X);
D = zeros(n+1,n);
N = zeros(.5*n*(n+1),n+1);
for np=1:n+1
    D(np,:) = (X(ModelIn(np),:)-X(xkin,:))/delta;
    N(:,np) = phi2eval(D(np,:))';
end

M = [ones(n+1,1) D]';
[Q,R] = qr(M');

% Now we add points until we have npmax starting with the most recent ones
i = nf;
while np<npmax
    D = (X(i,:)-X(xkin,:))/delta;
    if ~ismember(i,ModelIn) && norm(D)<=c2
        Ny = [N phi2eval(D)'];
        [Qy,Ry] = qrinsert(Q,R,np+1,[1 D],'row'); % Update QR factorization
        Ly = Ny*Qy(:,n+2:np+1);

        if (min(svd(Ly))>theta2)
            np = np+1;
            ModelIn(np,1) = i;
            N = Ny;
            Q = Qy;
            R = Ry;
            L = Ly;

            Z=Q(:,n+2:np);
            M=[M [1; D']]; % Note that M is growing
        end
    end

    i = i-1;
    if i==0
        % EVENTUALLY WILL NEED TO OUTPUT L AND M AND N AND Z IN THE EVENT THAT NP=N+1
        if np==(n+1)
            L=1; Z=zeros(n+1,.5*n*(n+1)); N=zeros(.5*n*(n+1),n+1); % set outputs so that hessian is zero
        end
        break
    end
end