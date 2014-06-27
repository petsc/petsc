function err = PetscFinalize()
%
%
err = calllib('libpetsc', 'PetscFinalize');PetscCHKERRQ(err);

