function [stage,err] = PetscLogStageRegister(stagename)
%
%  Registers a stage that can be logged with PetscLogStagePush()
%
[err,dummy,stage] = calllib('libpetsc', 'PetscLogStageRegister', stagename, 0);PetscCHKERRQ(err);


