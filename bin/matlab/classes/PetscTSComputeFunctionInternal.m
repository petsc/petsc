function err = PetscTSComputeFunctionInternal(pidts,time,pidx,pidxdot,pidf,funcname,ctx)
%
%   Used by TSComputeFunction_Matlab() to apply user Matlab function
%
%
err = 0;
x = PetscVec(pidx,'pobj');
xdot = PetscVec(pidxdot,'pobj');
f = PetscVec(pidf,'pobj');
ts = PetscTS(pidts,'pobj');
err = feval(funcname,ts,time,x,xdot,f,ctx);
