% function phi2eval modified 9/24/08 by Stefan Wild
%
% if x is a vec:
%   Phi=.5*[x(1)^2 sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n) x(2)^2 sqrt(2)*x(2)*x(3) ... x(n)^2]
% Assumes that X has points which are row vectors
%
% Used to be slower:
%for i=1:m
%    A = X(i,:)'*X(i,:)/sqrt(2);
%    A = A-(sqrt(2)-1)*diag(diag(A))/sqrt(2); % Need to do this to make sure diag is halved
%    Phi(i,1:.5*n*(n+1)) = A(logical(tril(ones(n))))'; % Put the lower triangular part in a vector
%end
function Phi = phi2eval(X)
[m,n] = size(X);
Phi = zeros(m,.5*n*(n+1));
j = 0;
for k=1:n
    j = j+1;
    Phi(:,j) = .5*X(:,k).^2;
    for kk=k+1:n
        j = j+1;
        Phi(:,j) = X(:,k).*X(:,kk)/sqrt(2);
    end
end