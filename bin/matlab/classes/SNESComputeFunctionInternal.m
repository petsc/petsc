function err = SNESComputeFunctionInternal(pidsnes,pidx,pidf,funcname,ctx)
%
%   Used by SNESComputeFunction_Matlab() to apply user Matlab function
%
%   Are pidx and pidf are the raw C pointers to the to Vecs
%
err = 0;
x = Vec(pidx,'pobj');
f = Vec(pidf,'pobj');
snes = SNES(pidsnes,'pobj');
err = feval(funcname,snes,x,f,ctx);
