function [flg,err] = nljacobian(snes,x,A,B,ctx)
%
%  Example of a nonlinear Jacobian needed by SNES
%
err = 0;
flg = Mat.SAME_NONZERO_PATTERN;
err = A.AssemblyBegin(Mat.FINAL_ASSEMBLY);
err = A.AssemblyEnd(Mat.FINAL_ASSEMBLY);
