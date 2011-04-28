function [err] = PetscSNESVIRedundancyCheckInternal(pidsnes,pidis_act,pidis_redact,funcname,ctx)
%
%   Used by vi->checkredundancy
%
%
err       = 0;
snes      = PetscSNES(pidsnes,'pobj');
isact     = PetscIS(pidis_act,'pobj');
is_redact = PetscIS(pidis_redact,'pobj');

[err,is_redact] = feval(funcname,snes,isact,is_redact,ctx);
