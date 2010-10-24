function [flg,err] = nljacobian(snes,x,A,B,ctx)
%
%  Example of a nonlinear Jacobian needed by SNES
%  Use identity as approximation for Jacobian
%
err = 0;
flg = Mat.SAME_NONZERO_PATTERN;
for i=0:9
  B.SetValues(i,i,1.0);
end
err = B.AssemblyBegin(Mat.FINAL_ASSEMBLY);
err = B.AssemblyEnd(Mat.FINAL_ASSEMBLY);
err = A.AssemblyBegin(Mat.FINAL_ASSEMBLY);
err = A.AssemblyEnd(Mat.FINAL_ASSEMBLY);
