function err = PetscSNESMonitorInternal(pidsnes,it,fnorm,pidx,funcname,ctx)
%
%   Used by SNESComputeFunction_Matlab() to apply user Matlab function
%
%
err = 0;
x = PetscVec(pidx,'pobj');
snes = PetscSNES(pidsnes,'pobj');
err = feval(funcname,snes,it,fnorm,x,ctx);
