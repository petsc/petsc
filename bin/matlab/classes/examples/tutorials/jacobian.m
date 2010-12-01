function [flg,err] = jacobian(dm,x,A,B)
%
%  Example of a Jacobian needed by DMSet Jacobian
%  Use identity as approximation for Jacobian
%
err = 0;
flg = PetscMat.SAME_NONZERO_PATTERN;
for i=0:15
  B.SetValues(i,i,1.0);
end
err = B.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
err = B.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);
err = A.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
err = A.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);
