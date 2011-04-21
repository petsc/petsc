function [err] = PetscSNESVIRedundancyCheckInternal(pidsnes,pidis_act,pidis_redact,funcname,ctx)
%
%   Used by SNESComputeJacobian_Matlab() to apply user Matlab Jacobian function
%
%
err       = 0;
snes      = PetscSNES(pidsnes,'pobj');
isact     = PetscIS(pidis_act,'pobj');
is_redact = PetscIS(pidis_redact,'pobj');

[err] = feval(funcname,snes,isact,is_redact,ctx);