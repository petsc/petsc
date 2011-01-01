function err = PetscLogStagePush
%
%  Start timing a stage
%
err = calllib('libpetsc', 'PetscLogStagePop');PetscCHKERRQ(err);


