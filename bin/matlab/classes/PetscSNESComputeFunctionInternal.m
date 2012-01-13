function err = PetscSNESComputeFunctionInternal(pidsnes,pidx,pidf,funcname,ctx)
%
%   Used by SNESComputeFunction_Matlab() to apply user Matlab function
%
%   Are pidx and pidf are the raw C pointers to the to Vecs
%
err = 0;
x = PetscVec(pidx,'pobj');
f = PetscVec(pidf,'pobj');
snes = PetscSNES(pidsnes,'pobj');
err = feval(funcname,snes,x,f,ctx);
