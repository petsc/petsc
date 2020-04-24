function x = dfoxs(n,nprob,factor)
%     This is a Matlab version of the subroutine dfoxs.f
%     This subroutine specifies the standard starting points for the
%     functions defined by subroutine dfovec as used in:
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
%     The subroutine returns
%     in x a multiple (factor) of the standard starting point. for
%     the 11th function the standard starting point is zero, so in
%     this case, if factor is not unity, then the subroutine returns
%     the vector  x(j) = factor, j=1,...,n.
%
%       xs is an output array of length n which contains the standard
%         starting point for problem nprob multiplied by factor.
%       n is a positive integer input variable.
%       nprob is a positive integer input variable which defines the
%         number of the problem. nprob must not exceed 22.
%       factor is an input variable which specifies the multiple of
%         the standard starting point. if factor is unity, no
%         multiplication is performed.
%
%     Argonne National Laboratory
%     Jorge More' and Stefan Wild. January 2008.

x = zeros(n,1);

switch nprob
    case 1 %     linear function - full rank or rank 1.
        x = ones(n,1);
    case 2 %     linear function - full rank or rank 1.
        x = ones(n,1);
    case 3 %     linear function - full rank or rank 1.
        x = ones(n,1);
    case 4 %     rosenbrock function.
        x(1) = -1.2;
        x(2) = 1;
    case 5 %     helical valley function.
        x(1) = -1;
    case 6 %     powell singular function.
        x(1) = 3;
        x(2) = -1;
        x(3) = 0;
        x(4) = 1;
    case 7 %     freudenstein and roth function.
        x(1) = .5;
        x(2) = -2;
    case 8 %     bard function.
        x(1:3) = 1;
    case 9 %     kowalik and osborne function.
        x(1) = .25;
        x(2) = .39;
        x(3) = .415;
        x(4) = .39;
    case 10 %     meyer function.
        x(1) = .02;
        x(2) = 4000;
        x(3) = 250;
    case 11 %     watson function.
        x = .5*ones(n,1);
    case 12 %     box 3-dimensional function.
        x(1) = 0;
        x(2) = 10;
        x(3) = 20;
    case 13 %     jennrich and sampson function.
        x(1) = .3;
        x(2) = .4;
    case 14 %     brown and dennis function.
        x(1) = 25;
        x(2) = 5;
        x(3) = -5;
        x(4) = -1;
    case 15 %     chebyquad function.
        for k = 1:n
            x(k) = k/(n+1);
        end
    case 16 %     brown almost-linear function.
        x = .5*ones(n,1);
    case 17 %     osborne 1 function.
        x(1) = .5;
        x(2) = 1.5;
        x(3) = 1;
        x(4) = .01;
        x(5) = .02;
    case 18 %     osborne 2 function.
        x(1) = 1.3;
        x(2) = .65;
        x(3) = .65;
        x(4) = .7;
        x(5) = .6;
        x(6) = 3;
        x(7) = 5;
        x(8) = 7;
        x(9) = 2;
        x(10) = 4.5;
        x(11) = 5.5;
    case 19 % bdqrtic
        x = ones(n,1);
    case 20 % cube
        x = .5*ones(n,1);
    case 21 % mancino
        for i=1:n
            ss = 0;
            for j=1:n
                ss = ss+(sqrt(i/j)*((sin(log(sqrt(i/j))))^5+(cos(log(sqrt(i/j))))^5));
            end
            x(i) = -8.710996D-4*((i-50)^3 + ss);
        end
    case 22  % Heart8
        x= [-.3, -.39, .3, -.344, -1.2, 2.69, 1.59, -1.5]';
end

x = factor*x;
