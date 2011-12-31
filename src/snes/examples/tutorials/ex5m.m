function localF = ex5m(localX,hx,hy,lambda)
% $Id: ex5m.m,v 1.3 2000/05/08 03:54:05 bsmith Exp $
%
%  Matlab routine that does the FormFunction() for ex5m.c
%
[m,n] = size(localX);
%
sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx;
%
%  copy over any potential boundary values
%
localF = localX;
%
%  compute interior u and derivatives
%
u   = localX(2:m-1,2:n-1);
uxx = (2.0*u - localX(1:m-2,2:n-1) - localX(3:m,2:n-1))*hydhx;
uyy = (2.0*u - localX(2:m-1,1:n-2) - localX(2:m-1,3:n))*hxdhy;
%
%  evaluate interior part of function
%
localF(2:m-1,2:n-1) = uxx + uyy - sc*exp(u);
%
%  This uses a clever (though not particularly efficient) way of 
% evaluating the function so that it works for any subdomain
% (with or without any of the true boundary)