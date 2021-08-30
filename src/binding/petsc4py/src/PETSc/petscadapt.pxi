cdef extern from * nogil:
    int TSGetAdapt(PetscTS,PetscTSAdapt*)
    int TSAdaptGetStepLimits(PetscTSAdapt,PetscReal*,PetscReal*)
    int TSAdaptSetStepLimits(PetscTSAdapt,PetscReal,PetscReal)
