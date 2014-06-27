function err = nlfunction(snes,x,f,ctx)
%
%  Example of a nonlinear function needed by SNES
%
err = 0;
f(:) = .5*x(:).*x(:) - 1;
