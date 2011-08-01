function [flg,err] = nljacobian(snes,x,A,B,ctx)
%
%  Example of a nonlinear Jacobian needed by SNES
%
err = 0;
flg = PetscMat.SAME_NONZERO_PATTERN;
err = A.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
err = A.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);
