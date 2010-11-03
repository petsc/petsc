function [flg,err] = PetscTSComputeJacobianInternal(pidts,time,pidx,pidxdot,shift,pidA,pidB,funcname,ctx)
%
%   Used by TSComputeJacobian_Matlab() to apply user Matlab Jacobian function
%
%
err = 0;
x = PetscVec(pidx,'pobj');
xdot = PetscVec(pidxdot,'pobj');
A = PetscMat(pidA,'pobj');
B = PetscMat(pidB,'pobj');
ts = PetscSNES(pidts,'pobj');
[flg,err] = feval(funcname,ts,time,x,xdot,shift,A,B,ctx);
