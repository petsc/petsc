function [err,pidis_redact] = PetscSNESVIRedundancyCheckInternal(pidsnes,pidis_act,funcname,ctx)
%
%   Used by SNESComputeJacobian_Matlab() to apply user Matlab Jacobian function
%
%
err = 0;
isact = PetscIS(pidis_act,'pobj');
snes = PetscSNES(pidsnes,'pobj');
[is_redact,err] = feval(funcname,snes,isact,ctx);
pidis_redact = [];
if(is_redact)
    pidis_redact = is_redact.pobj;
end