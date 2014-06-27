function [str,err] = PetscDMComputeJacobianInternal(piddm,pidx,pidA,pidB,funcname)
%
%   Used by DMComputeJacobian_Matlab() to apply user Matlab function
%
err = 0;
x = PetscVec(pidx,'pobj');
A = PetscMat(pidA,'pobj');
B = PetscMat(pidB,'pobj');
dm = PetscDM(piddm,'pobj');
[str,err] = feval(funcname,dm,x,A,B);
