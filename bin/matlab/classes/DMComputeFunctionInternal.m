function err = DMComputeFunctionInternal(piddm,pidx,pidf,funcname)
%
%   Used by DMComputeFunction_Matlab() to apply user Matlab function
%
%   pidx and pidf are the raw C pointers to the to Vecs
%
err = 0;
x = Vec(pidx,'pobj');
f = Vec(pidf,'pobj');
dm = DM(piddm,'pobj');
err = feval(funcname,dm,x,f);
