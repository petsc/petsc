function hl = data_profile(H,N,gate)
%     This subroutine produces a data profile as described in:
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
%     The subroutine returns a handle to lines in a data profile.
%
%       H contains a three dimensional array of function values.
%         H(f,p,s) = function value # f for problem p and solver s.
%       N is an np-by-1 vector of (positive) budget units. If simplex
%         gradients are desired, then N(p) would be n(p)+1, where n(p) is
%         the number of variables for problem p.
%       gate is a positive constant reflecting the convergence tolerance.
%
%     Argonne National Laboratory
%     Jorge More' and Stefan Wild. January 2008.

[nf,np,ns] = size(H); % Grab the dimensions

% Produce a suitable history array with sorted entries:
for j = 1:ns
    for i = 2:nf
      H(i,:,j) = min(H(i,:,j),H(i-1,:,j));
    end
end

prob_min = min(min(H),[],3);   % The global minimum for each problem
prob_max = H(1,:,1);           % The starting value for each problem

% For each problem and solver, determine the number of
% N-function bundles (e.g.- gradients) required to reach the cutoff value
T = zeros(np,ns);
for p = 1:np
    cutoff = prob_min(p) + gate*(prob_max(p) - prob_min(p));
    for s = 1:ns
        nfevs = find(H(:,p,s) <= cutoff,1);
        if (isempty(nfevs))
            T(p,s) = NaN;
        else
            T(p,s) = nfevs/N(p);
        end
    end
end

% Other colors, lines, and markers are easily possible:
colors  = ['b' 'r' 'k' 'm' 'c' 'g' 'y'];   lines   = {'-' '-.' '--'};
markers = [ 's' 'o' '^' 'v' 'p' '<' 'x' 'h' '+' 'd' '*' '<' ];

% Replace all NaN's with twice the max_ratio and sort.
max_data = max(max(T));
T(isnan(T)) = 2*max_data;
T = sort(T);

% For each solver, plot stair graphs with markers.
hl = zeros(ns,1);
for s = 1:ns
    [xs,ys] = stairs(T(:,s),(1:np)/np);
    sl = mod(s-1,3) + 1; sc = mod(s-1,7) + 1; sm = mod(s-1,12) + 1;
    option1 = [char(lines(sl)) colors(sc) markers(sm)];
    
    hl(s) = plot(xs,ys,option1);
    hold on;
end

% Axis properties are set so that failures are not shown, but with the
% max_ratio data points shown. This highlights the "flatline" effect.
axis([0 1.1*max_data 0 1]);
