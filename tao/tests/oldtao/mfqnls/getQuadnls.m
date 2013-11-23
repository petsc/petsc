function [cres,Gres,Hres,G,H] = getQuadnls(L,Z,M,N,F,Fres,F0,mtype,m)
% Updated 9/23/08 by Stefan Wild
% Computes the parameters of the quadratic Q(x) = c + g'*x + 0.5*x*G*x
% that satisfies the interpolation conditions Q(X[:,j]) = f(j)
% for j = 1,...,m and with a Hessian matrix of least Frobenius norm.
n = size(M,1)-1;
Hres = zeros(n);
if mtype==1
    G = zeros(n,m); 
    H = zeros(n,n,m);
    for k=1:m
        % For L=N*Z, solve L'*L*Omega = Z'*f:
        Omega = L'\(Z'*F(:,k));
        Omega = L \ Omega;
        Beta = L*Omega;
        Alpha = M'\(F(:,k)-N'*Beta);

        G(:,k) = Alpha(2:n+1);
        num = 0;
        for i=1:n
            num = num+1;
            H(i,i,k) = Beta(num);
            for j=i+1:n
                num = num+1;
                H(i,j,k) = Beta(num)/sqrt(2);
                H(j,i,k) = H(i,j,k);
            end
        end
    end

    Gres = G*F0(1:m)';
    for i=1:m
        Hres = Hres + F0(i)*H(:,:,i);
    end
    Hres = Hres + G*G';
else
    % For L=N*Z, solve L'*L*Omega = Z'*f:
    Omega = L'\(Z'*Fres);
    Omega = L \ Omega;
    Beta = L*Omega;
    Alpha = M'\(Fres-N'*Beta);

    Gres = Alpha(2:n+1);
    num = 0;
    for i=1:n
        num = num+1;
        Hres(i,i) = Beta(num);
        for j=i+1:n
            num = num+1;
            Hres(i,j) = Beta(num)/sqrt(2);
            Hres(j,i) = Hres(i,j);
        end
    end
end
cres = Alpha(1); % Currently not really needed/useful, should be revisited 