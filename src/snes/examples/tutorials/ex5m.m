function localF = ex5m(localX,hx,hy,lambda)
%
%  Matlab routine that does the FormFunction() for ex5.c
%  when using the Matlab engine
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
