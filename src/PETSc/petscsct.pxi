# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef char* PetscScatterType "const char*"
    PetscScatterType VECSCATTERSEQ
    PetscScatterType VECSCATTERMPI1
    PetscScatterType VECSCATTERMPI3
    PetscScatterType VECSCATTERMPI3NODE
    PetscScatterType VECSCATTERSF

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
