# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef const char* PetscScatterType "VecScatterType"
    PetscScatterType PETSCSFBASIC
    PetscScatterType PETSCSFNEIGHBOR
    PetscScatterType PETSCSFALLGATHERV
    PetscScatterType PETSCSFALLGATHER
    PetscScatterType PETSCSFGATHERV
    PetscScatterType PETSCSFGATHER
    PetscScatterType PETSCSFALLTOALL
    PetscScatterType PETSCSFWINDOW

    int VecScatterView(PetscScatter,PetscViewer)
    int VecScatterDestroy(PetscScatter*)
    int VecScatterSetUp(PetscScatter)
    int VecScatterCreate(PetscVec,PetscIS,PetscVec,PetscIS,PetscScatter*)
    int VecScatterSetFromOptions(PetscScatter)
    int VecScatterSetType(PetscScatter,PetscScatterType)
    int VecScatterGetType(PetscScatter,PetscScatterType*)
    int VecScatterCopy(PetscScatter, PetscScatter*)
    int VecScatterCreateToAll(PetscVec,PetscScatter*,PetscVec*)
    int VecScatterCreateToZero(PetscVec,PetscScatter*,PetscVec*)
    int VecScatterBegin(PetscScatter,PetscVec,PetscVec,PetscInsertMode,PetscScatterMode)
    int VecScatterEnd(PetscScatter,PetscVec,PetscVec,PetscInsertMode,PetscScatterMode)

# --------------------------------------------------------------------
