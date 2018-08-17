function fvec = dfovec(m,n,x,nprob)
%     This is a Matlab version of the subroutine dfovec.f
%     This subroutine specifies the nonlinear benchmark problems in
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
%     The data file dfo.dat defines suitable values of m and n
%     for each problem number nprob.
%
%     This subroutine defines the functions of 22 nonlinear
%     least squares problems. The allowable values of (m,n) for
%     functions 1,2 and 3 are variable but with m .ge. n.
%     For functions 4,5,6,7,8,9 and 10 the values of (m,n) are
%     (2,2),(3,3),(4,4),(2,2),(15,3),(11,4) and (16,3), respectively.
%     Function 11 (Watson) has m = 31 with n usually 6 or 9.
%     However, any n, n = 2,...,31, is permitted.
%     Functions 12,13 and 14 have n = 3,2 and 4, respectively, but
%     allow any m .ge. n, with the usual choices being 10,10 and 20.
%     Function 15 (Chebyquad) allows m and n variable with m .ge. n.
%     Function 16 (Brown) allows n variable with m = n.
%     For functions 17 and 18, the values of (m,n) are
%     (33,5) and (65,11), respectively.
%
%   fvec = ssqfcn(m,n,x,nprob)
%       fvec is an output array of length m which contains the nprob
%         function evaluated at x.
%       m and n are positive integer input variables. n must not
%         exceed m.
%       x is an input array of length n.
%       nprob is a positive integer input variable which defines the
%         number of the problem. nprob must not exceed 22.
%
%     Argonne National Laboratory
%     Jorge More' and Stefan Wild. January 2008.

% Set lots of constants:
c13 = 1.3d1; c14 = 1.4d1; c29 = 2.9d1; c45 = 4.5d1;
v  = [4.0d0,2.0d0,1.0d0,5.0d-1,2.5d-1,1.67d-1,1.25d-1,1.0d-1,8.33d-2,...
    7.14d-2,6.25d-2];
y1 = [1.4d-1,1.8d-1,2.2d-1,2.5d-1,2.9d-1,3.2d-1,3.5d-1,3.9d-1,3.7d-1,...
    5.8d-1,7.3d-1,9.6d-1,1.34d0,2.1d0,4.39d0];
y2 = [1.957d-1,1.947d-1,1.735d-1,1.6d-1,8.44d-2,6.27d-2,4.56d-2,3.42d-2,...
    3.23d-2,2.35d-2,2.46d-2];
y3 = [3.478d4,2.861d4,2.365d4,1.963d4,1.637d4,1.372d4,1.154d4,9.744d3,...
    8.261d3,7.03d3,6.005d3,5.147d3,4.427d3,3.82d3,3.307d3,2.872d3];
y4 = [8.44d-1,9.08d-1,9.32d-1,9.36d-1,9.25d-1,9.08d-1,8.81d-1,8.5d-1,...
    8.18d-1,7.84d-1,7.51d-1,7.18d-1,6.85d-1,6.58d-1,6.28d-1,6.03d-1,...
    5.8d-1,5.58d-1,5.38d-1,5.22d-1,5.06d-1,4.9d-1,4.78d-1,4.67d-1,...
    4.57d-1,4.48d-1,4.38d-1,4.31d-1,4.24d-1,4.2d-1,4.14d-1,4.11d-1,...
    4.06d-1];
y5 = [1.366d0,1.191d0,1.112d0,1.013d0,9.91d-1,8.85d-1,8.31d-1,8.47d-1,...
    7.86d-1,7.25d-1,7.46d-1,6.79d-1,6.08d-1,6.55d-1,6.16d-1,6.06d-1,...
    6.02d-1,6.26d-1,6.51d-1,7.24d-1,6.49d-1,6.49d-1,6.94d-1,6.44d-1,...
    6.24d-1,6.61d-1,6.12d-1,5.58d-1,5.33d-1,4.95d-1,5.0d-1,4.23d-1,...
    3.95d-1,3.75d-1,3.72d-1,3.91d-1,3.96d-1,4.05d-1,4.28d-1,4.29d-1,...
    5.23d-1,5.62d-1,6.07d-1,6.53d-1,6.72d-1,7.08d-1,6.33d-1,6.68d-1,...
    6.45d-1,6.32d-1,5.91d-1,5.59d-1,5.97d-1,6.25d-1,7.39d-1,7.1d-1,...
    7.29d-1,7.2d-1,6.36d-1,5.81d-1,4.28d-1,2.92d-1,1.62d-1,9.8d-2,5.4d-2];

% Initialize things:
fvec = zeros(m,1);
sum = 0;

switch nprob
    case 1 % Linear function - full rank.
        for j = 1:n
            sum = sum + x(j);
        end
        temp = 2*sum/m + 1;
        for i = 1:m
            fvec(i) = -temp;
            if (i <= n)
                fvec(i) = fvec(i) + x(i);
            end
        end
    case 2 %     Linear function - rank 1.
        for j = 1:n
            sum = sum + j*x(j);
        end
        for i = 1:m
            fvec(i) = i*sum - 1;
        end
    case 3 %     Linear function - rank 1 with zero columns and rows.
        for j = 2:n-1
            sum = sum + j*x(j);
        end
        for i = 1:m-1
            fvec(i) = (i-1)*sum - 1;
        end
        fvec(m) = -1;
    case 4 %     Rosenbrock function.
        fvec(1) = 10*(x(2) - x(1)^2);
        fvec(2) = 1 - x(1);
    case 5 %     Helical valley function.
        if (x(1) > 0)
            th = atan(x(2)/x(1))/(2*pi);
        elseif (x(1) < 0)
            th = atan(x(2)/x(1))/(2*pi) + .5;
        else    % x(1)=0
            th = .25;
        end
        r = sqrt(x(1)^2+x(2)^2);
        fvec(1) = 10*(x(3) - 10*th);
        fvec(2) = 10*(r-1);
        fvec(3) = x(3);
    case 6 %     Powell singular function.
        fvec(1) = x(1) + 10*x(2);
        fvec(2) = sqrt(5)*(x(3) - x(4));
        fvec(3) = (x(2) - 2*x(3))^2;
        fvec(4) = sqrt(10)*(x(1) - x(4))^2;
    case  7 %     Freudenstein and Roth function.
        fvec(1) = -c13 + x(1) + ((5 - x(2))*x(2) - 2)*x(2);
        fvec(2) = -c29 + x(1) + ((1 + x(2))*x(2) - c14)*x(2);
    case 8 %     Bard function.
        for i = 1:15
            tmp1 = i;
            tmp2 = 16-i;
            tmp3 = tmp1;
            if (i > 8)
                tmp3 = tmp2;
            end
            fvec(i) = y1(i) - (x(1) + tmp1/(x(2)*tmp2 + x(3)*tmp3));
        end
    case 9 %     Kowalik and Osborne function.
        for i = 1:11
            tmp1 = v(i)*(v(i) + x(2));
            tmp2 = v(i)*(v(i) + x(3)) + x(4);
            fvec(i) = y2(i) - x(1)*tmp1/tmp2;
        end
    case 10 %     Meyer function.
        for i = 1:16
            temp = 5*i + c45 + x(3);
            tmp1 = x(2)/temp;
            tmp2 = exp(tmp1);
            fvec(i) = x(1)*tmp2 - y3(i);
        end
    case 11 %     Watson function.
        for i = 1:29
            div = i/c29;
            s1 = 0;
            dx = 1;
            for j = 2:n
                s1 = s1 + (j-1)*dx*x(j);
                dx = div*dx;
            end
            s2 = 0;
            dx = 1;
            for j = 1:n
                s2 = s2 + dx*x(j);
                dx = div*dx;
            end
            fvec(i) = s1 - s2^2 - 1;
        end
        fvec(30) = x(1);
        fvec(31) = x(2) - x(1)^2 - 1;
    case 12 %     Box 3-dimensional function.
        for i = 1:m
            temp = i;
            tmp1 = temp/10;
            fvec(i) = exp(-tmp1*x(1)) - exp(-tmp1*x(2))+ ...
                (exp(-temp) - exp(-tmp1))*x(3);
        end
    case 13 %     Jennrich and Sampson function.
        for i = 1:m
            temp = i;
            fvec(i) = 2 + 2*temp - exp(temp*x(1)) - exp(temp*x(2));
        end
    case 14 %     Brown and Dennis function.
        for i = 1:m
            temp = i/5;
            tmp1 = x(1) + temp*x(2) - exp(temp);
            tmp2 = x(3) + sin(temp)*x(4) - cos(temp);
            fvec(i) = tmp1^2 + tmp2^2;
        end
    case 15 %     Chebyquad function.
        for j = 1:n
            t1 = 1;
            t2 = 2*x(j) - 1;
            t = 2*t2;
            for i = 1:m
                fvec(i) = fvec(i) + t2;
                th = t*t2 - t1;
                t1 = t2;
                t2 = th;
            end
        end
        iev = -1;
        for i = 1:m
            fvec(i) = fvec(i)/n;
            if (iev > 0)
                fvec(i) = fvec(i) + 1/(i^2 - 1);
            end
            iev = -iev;
        end
    case 16 %     Brown almost-linear function.
        sum1 = -(n+1);
        prod1 = 1;
        for j = 1:n
            sum1 = sum1 + x(j);
            prod1 = x(j)*prod1;
        end
        for i = 1:n-1
            fvec(i) = x(i) + sum1;
        end
        fvec(n) = prod1 - 1;
    case 17 %     Osborne 1 function.
        for i = 1:33
            temp = 10*(i-1);
            tmp1 = exp(-x(4)*temp);
            tmp2 = exp(-x(5)*temp);
            fvec(i) = y4(i) - (x(1) + x(2)*tmp1 + x(3)*tmp2);
        end
    case 18 %     Osborne 2 function.
        for i = 1:65
            temp = (i-1)/10;
            tmp1 = exp(-x(5)*temp);
            tmp2 = exp(-x(6)*(temp-x(9))^2);
            tmp3 = exp(-x(7)*(temp-x(10))^2);
            tmp4 = exp(-x(8)*(temp-x(11))^2);
            fvec(i) = y5(i) - (x(1)*tmp1 + x(2)*tmp2 + ...
            x(3)*tmp3 + x(4)*tmp4);
        end
    case 19 % Bdqrtic
        % n>=5, m = (n-4)*2
        for i=1:n-4
            fvec(i)=(-4*x(i)+3.0);
            fvec(n-4+i)=(x(i)^2+2*x(i+1)^2+3*x(i+2)^2+4*x(i+3)^2+5*x(n)^2);
        end
    case 20 % Cube
        % n=2; m=n;
        fvec(1) = (x(1)-1.0);
        for i=2:n
        		fvec(i) = 10*(x(i)-x(i-1)^3);
        end       
    case 21 % Mancino
        % n >=2; m=n
        for i=1:n
            ss=0;
            for j=1:n
                v2 = sqrt (x(i)^2 +i/j);
                ss = ss+v2*((sin(log(v2)))^5 + (cos(log(v2)))^5);
            end
            fvec(i)=1400*x(i) + (i-50)^3 + ss;
        end
    case 22 % Heart8ls
        % m=n=8
        fvec(1) = x(1) + x(2) + 0.69;
        fvec(2) = x(3) + x(4) + 0.044;
        fvec(3) = x(5)*x(1) + x(6)*x(2) - x(7)*x(3) - x(8)*x(4) + 1.57;
        fvec(4) = x(7)*x(1) + x(8)*x(2) + x(5)*x(3) + x(6)*x(4) + 1.31;
        fvec(5) = x(1)*(x(5)^2-x(7)^2) - 2.0*x(3)*x(5)*x(7) + ...
            x(2)*(x(6)^2-x(8)^2) - 2.0*x(4)*x(6)*x(8) + 2.65;
        fvec(6) = x(3)*(x(5)^2-x(7)^2) + 2.0*x(1)*x(5)*x(7) + ...
            x(4)*(x(6)^2-x(8)^2) + 2.0*x(2)*x(6)*x(8) - 2.0;
        fvec(7) = x(1)*x(5)*(x(5)^2-3.0*x(7)^2) + ...
            x(3)*x(7)*(x(7)^2-3.0*x(5)^2) + ...
            x(2)*x(6)*(x(6)^2-3.0*x(8)^2) + ...
            x(4)*x(8)*(x(8)^2-3.0*x(6)^2) + 12.6;
        fvec(8) = x(3)*x(5)*(x(5)^2-3.0*x(7)^2) - ...
            x(1)*x(7)*(x(7)^2-3.0*x(5)^2) + ...
            x(4)*x(6)*(x(6)^2-3.0*x(8)^2) - ...
            x(2)*x(8)*(x(8)^2-3.0*x(6)^2) - 9.48;
 
end