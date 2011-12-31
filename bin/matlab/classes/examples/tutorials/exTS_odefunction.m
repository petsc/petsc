function err = exTS_odefunction(ts,time,x,xdot,f,ctx)
%
%  Example of a function needed by TS
%
err = 0;
f(:) = xdot(:) + x(:);
