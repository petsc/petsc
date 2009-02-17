cdef extern from "petsc.h" nogil:

    ctypedef double PetscLogDouble

    int PetscLogFlops(PetscLogDouble)
    int PetscGetFlops(PetscLogDouble*)

    int PetscGetTime(PetscLogDouble*)
    int PetscGetCPUTime(PetscLogDouble*)

    int PetscMallocGetCurrentUsage(PetscLogDouble*)
    int PetscMemoryGetCurrentUsage(PetscLogDouble*)
