function y = calfun(x)
%     This is a Matlab version of the subroutine calfun.f
%     This subroutine returns a function value as used in:
%
%     Benchmarking Derivative-Free Optimization Algorithms
%     Jorge J. More' and Stefan M. Wild
%     SIAM J. Optimization, Vol. 20 (1), pp.172-191, 2009.
%
%     The latest version of this subroutine is always available at
%     http://www.mcs.anl.gov/~more/dfo/
%     The authors would appreciate feedback and experiences from numerical
%     studies conducted using this subroutine.
%
%     The subroutine returns the function value f(x)
%
%       x is an input array of length n.
%       f is an output that contains the function value at x.
%
%     The rand generator should be seeded before this is called
%
%     Additional problem descriptors are passed through the global
%     variables:
%       m a positive integer (length of output from dfovec).
%          m must not exceed n.
%       nprob is a positive integer that defines the number of the problem.
%          nprob must not exceed 22.
%       probtype is a string specifying the type of problem desired:
%           'smooth' corresponds to smooth problems
%           'nondiff' corresponds to piecewise-smooth problems
%           'wild3' corresponds to deterministically noisy problems
%           'noisy3' corresponds to stochastically noisy problems
%
%     To store the evaluation history, additional variables are passed 
%     through global variables. These may be commented out if a user 
%     desires. They are:
%       nfev is a non-negative integer containing the number of function 
%          evaluations done so far (nfev=0 is a good default).
%          after calling calfun, nfev will be incremented by one.
%       np is a counter for the test problem number. np=1 is a good
%          default if only a single problem/run will be done.
%       fvals is a matrix containing the history of function
%          values, the entry fvals(nfev+1,np) being updated here.
%
%     Argonne National Laboratory
%     Jorge More' and Stefan Wild. January 2008.

global m nprob probtype fvals fvecs nfev np sigma X_hist 
x = x(:);

n = size(x,1); % Problem dimension

% Restrict domain for some nondiff problems
xc = x;
if strcmp('nondiff',probtype)
    if nprob==8 || nprob==9 || nprob==13 || nprob==16 || nprob==17 || nprob==18
        xc = max(x,0);
    end
end

% Generate the vector
fvec = dfovec(m,n,xc,nprob); 

% J = jacobian(m,n,x,nprob);
% grad = J'*sign(fvec);
% disp(grad)

% Calculate the function value
switch probtype
    case 'noisy3'
        sigma=10^-3;
        u = sigma*(-ones(m,1)+2*rand(m,1));
        fvec = fvec.*(1+u);
        y = sum(fvec.^2);
    case 'wild3'
        sigma=10^-3;
        phi = 0.9*sin(100*norm(x,1))*cos(100*norm(x,inf)) + 0.1*cos(norm(x,2));
        phi = phi*(4*phi^2 -3);
        y = (1 + sigma*phi)*sum(fvec.^2);
    case 'smooth'
        y = sum(fvec.^2);
    case 'nondiff'
        y = sum(abs(fvec));
    % for the sake of experiments on nonsmooth noisy functions
    % sigma should be declared as a global variable
    case 'ndnoisy3'
        u = sigma*randn(m,1);
        fvec = fvec + u;
        y = sum(abs(fvec));
end

% Update the function value history
nfev = nfev +1;
fvecs(nfev,:) = fvec;
fvals(nfev,:) = y;
X_hist(nfev,:) = x';

% flag = 1;
% if flag && nfev == 1 && n == 2
%     if exist(['./' num2str(nprob) '_plot_points.mat'],'file')
%         load(['./' num2str(nprob) '_plot_points.mat'])
%         hold off
%     else
%     inc = 0.025;
%     sigma = 0;
%     [X,Y] = meshgrid(-2:inc:2,-2:inc:2);
%     Z = zeros(size(X));
%     for i = 1:size(X,1); 
%         for j = 1:size(X,2); 
%             Z(i,j) = calfun([X(i,j),Y(i,j)]); 
%         end; 
%         disp(i); 
%     end;
%     save(['./' num2str(nprob) '_plot_points.mat'],'X','Y','Z')    
%     error('DONE')
%     end
%     contour(X,Y,log2(Z-min(min(Z))+1),20,'linewidth',2)
%     hold on
%     axis square    
% end
% if flag && n == 2
%     scatter(x(1),x(2),80,'b','filled');
%     pause
% end

end
