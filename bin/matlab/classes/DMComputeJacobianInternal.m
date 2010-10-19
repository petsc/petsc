function [str,err] = DMComputeJacobianInternal(piddm,pidx,pidA,pidB,funcname)
%
%   Used by DMComputeJacobian_Matlab() to apply user Matlab function
%
err = 0;
x = Vec(pidx,'pobj');
A = Mat(pidA,'pobj');
B = Mat(pidB,'pobj');
dm = DM(piddm,'pobj');
[str,err] = feval(funcname,dm,x,A,B);
