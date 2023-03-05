cdef extern from * nogil:

    ctypedef const char* PetscRandomType
    PetscRandomType PETSCRAND
    PetscRandomType PETSCRAND48
    PetscRandomType PETSCSPRNG
    PetscRandomType PETSCRANDER48
    PetscRandomType PETSCRANDOM123

    PetscErrorCode PetscRandomCreate(MPI_Comm,PetscRandom*)
    PetscErrorCode PetscRandomDestroy(PetscRandom*)
    PetscErrorCode PetscRandomView(PetscRandom,PetscViewer)

    PetscErrorCode PetscRandomSetType(PetscRandom,PetscRandomType)
    PetscErrorCode PetscRandomGetType(PetscRandom,PetscRandomType*)
    PetscErrorCode PetscRandomSetFromOptions(PetscRandom)

    PetscErrorCode PetscRandomGetValue(PetscRandom,PetscScalar*)
    PetscErrorCode PetscRandomGetValueReal(PetscRandom,PetscReal*)
    PetscErrorCode PetscRandomGetValueImaginary(PetscRandom,PetscScalar*)
    PetscErrorCode PetscRandomGetInterval(PetscRandom,PetscScalar*,PetscScalar*)
    PetscErrorCode PetscRandomSetInterval(PetscRandom,PetscScalar,PetscScalar)
    PetscErrorCode PetscRandomSetSeed(PetscRandom,unsigned long)
    PetscErrorCode PetscRandomGetSeed(PetscRandom,unsigned long*)
    PetscErrorCode PetscRandomSeed(PetscRandom)
