function comm = PETSC_COMM_SELF()
[err,comm] = calllib('libpetsc', 'PetscGetPETSC_COMM_SELF',0);PetscCHKERRQ(err);
