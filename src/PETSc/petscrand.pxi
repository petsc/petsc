cdef extern from * nogil:

    ctypedef char* PetscRandomType "const char*"
    PetscRandomType PETSCRAND
    PetscRandomType PETSCRAND48
    PetscRandomType PETSCSPRNG

    int PetscRandomCreate(MPI_Comm,PetscRandom*)
    int PetscRandomDestroy(PetscRandom*)
    int PetscRandomView(PetscRandom,PetscViewer)

    int PetscRandomSetType(PetscRandom,PetscRandomType)
    int PetscRandomGetType(PetscRandom,PetscRandomType*)
    int PetscRandomSetFromOptions(PetscRandom)

    int PetscRandomGetValue(PetscRandom,PetscScalar*)
    int PetscRandomGetValueReal(PetscRandom,PetscReal*)
    int PetscRandomGetValueImaginary(PetscRandom,PetscScalar*)
    int PetscRandomGetInterval(PetscRandom,PetscScalar*,PetscScalar*)
    int PetscRandomSetInterval(PetscRandom,PetscScalar,PetscScalar)
    int PetscRandomSetSeed(PetscRandom,unsigned long)
    int PetscRandomGetSeed(PetscRandom,unsigned long*)
    int PetscRandomSeed(PetscRandom)
