function err = nlfunction(snes,x,f,ctx)
%
%  Example of a nonlinear function needed by SNES
%
err = 0;
f(:) = x(:).*x(:) - 1;
%err =  x.Copy(f);
