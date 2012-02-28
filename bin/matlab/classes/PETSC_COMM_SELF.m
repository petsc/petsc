function comm = PETSC_COMM_SELF()
[err,comm] = calllib('libpetsc', 'PetscGetPETSC_COMM_SELFMatlab',0);PetscCHKERRQ(err);
