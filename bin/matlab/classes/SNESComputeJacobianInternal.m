function [flg,err] = SNESComputeJacobianInternal(pidsnes,pidx,pidA,pidB,funcname,ctx)
%
%   Used by SNESComputeJacobian_Matlab() to apply user Matlab Jacobian function
%
%
err = 0;
x = Vec(pidx,'pobj');
A = Mat(pidA,'pobj');
B = Mat(pidB,'pobj');
snes = SNES(pidsnes,'pobj');
[flg,err] = feval(funcname,snes,x,A,B,ctx);
