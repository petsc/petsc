function err = PetscDMComputeFunctionInternal(piddm,pidx,pidf,funcname)
%
%   Used by DMComputeFunction_Matlab() to apply user Matlab function
%
%   pidx and pidf are the raw C pointers to the to Vecs
%
err = 0;
x = PetscVec(pidx,'pobj');
f = PetscVec(pidf,'pobj');
dm = PetscDM(piddm,'pobj');
err = feval(funcname,dm,x,f);
