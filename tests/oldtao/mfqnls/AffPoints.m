% AffPoints.m                            Created by Stefan Wild & Jorge More'
% Version 0.0,    Modified 1/8/08
% This routine finds the affinely indep points used by MNH.
%
% INPUTS-------------------------------------------------------------------
% X       [dbl] [nf-by-n] Stores the evaluation point locations
% Q       [dbl] [mm-by-mm]  %** look into
% R       [dbl] [mm-by-n+1] %** look into
% ModelIn [int] [np-by-1] Integer vector of model interpolation indices
% xkin    [int] Index of current center
% delta   [dbl] Positive trust region radius
% theta1  [dbl] Pivot threshold 
% c1      [dbl] Factor for checking validity
%
% OUTPUTS------------------------------------------------------------------
% Modeld  [dbl] [1-by-n]  Unit direction to improve model
%
function [ModelIn,Modeld,Q,R] = AffPoints(X,Q,R,ModelIn,xkin,delta,theta1,c1)
% --INTERMEDIATE-----------------------------------------------------------
% D       [dbl] [np-by-n] Stores the evaluation point displacements
% n       [int] Dimension (number of continuous variables)
% nf      [int] Counter for the number of function evaluations
% np      [int] Number of model interpolation points
% proj    [dbl] Value of the appropriate projection
% -------------------------------------------------------------------------

[nf,n] = size(X)
np = size(ModelIn,1) % Stores (<=n) indices of the interpolation points
%Modeld = zeros(1,n); % Initialize for output
for i = nf:-1:1
    D = (X(i,:)-X(xkin,:))/delta;
    normD = norm(D)
    if norm(D)<= c1
        showD = D
        proj = norm(D*Q(:,np+1:n),2) % Project D onto null
        if (proj>=theta1) % add this index to ModelIn
            np = np+1;
            ModelIn(np,1) = i;
            [Q,R] = qrinsert(Q,R,np,D'); % Update QR factorization
            if (np==n)
                break; % Breaks out of for loop
            end
        end
    end
end

Modeld = Q(:,np+1:n)'  % Will be empty if np=n
size(ModelIn)

%%% Eventually note that don't need to output both Q and Modeld