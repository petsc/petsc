cdef extern from "petsc.h":

    ctypedef double PetscLogDouble

    int PetscLogFlops(PetscLogDouble)
    int PetscGetFlops(PetscLogDouble*)

    int PetscGetTime(PetscLogDouble*)
    int PetscGetCPUTime(PetscLogDouble*)
