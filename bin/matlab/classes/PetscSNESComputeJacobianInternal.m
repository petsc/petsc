function [flg,err] = PetscSNESComputeJacobianInternal(pidsnes,pidx,pidA,pidB,funcname,ctx)
%
%   Used by SNESComputeJacobian_Matlab() to apply user Matlab Jacobian function
%
%
err = 0;
x = PetscVec(pidx,'pobj');
A = PetscMat(pidA,'pobj');
B = PetscMat(pidB,'pobj');
snes = PetscSNES(pidsnes,'pobj');
[flg,err] = feval(funcname,snes,x,A,B,ctx);
