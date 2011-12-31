function err = PetscLogStagePush(stage)
%
%  Start timing a stage
%
err = calllib('libpetsc', 'PetscLogStagePush', stage);PetscCHKERRQ(err);


