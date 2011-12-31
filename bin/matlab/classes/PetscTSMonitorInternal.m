function err = PetscTSMonitorInternal(pidts,it,time,pidx,funcname,ctx)
%
%   Used by TSComputeFunction_Matlab() to apply user Matlab function
%
%
err = 0;
x = PetscVec(pidx,'pobj');
ts = PetscTS(pidts,'pobj');
err = feval(funcname,ts,it,time,x,ctx);
