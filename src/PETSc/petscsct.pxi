# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef char* PetscScatterType "const char*"
    PetscScatterType SCATTERSEQ      "VECSCATTERSEQ"
    PetscScatterType SCATTERMPI1     "VECSCATTERMPI1"
    PetscScatterType SCATTERMPI3     "VECSCATTERMPI3"
    PetscScatterType SCATTERMPI3NODE "VECSCATTERMPI3NODE"

    int VecScatterView(PetscScatter,PetscViewer)
    int VecScatterDestroy(PetscScatter*)
    int VecScatterCreate(PetscVec,PetscIS,PetscVec,PetscIS,PetscScatter*)
    int VecScatterCreateEmpty(MPI_Comm,PetscScatter*)
    int VecScatterSetFromOptions(PetscScatter)
    int VecScatterSetType(PetscScatter,PetscScatterType)
    int VecScatterGetType(PetscScatter,PetscScatterType*)
    int VecScatterCopy(PetscScatter, PetscScatter*)
    int VecScatterCreateToAll(PetscVec,PetscScatter*,PetscVec*)
    int VecScatterCreateToZero(PetscVec,PetscScatter*,PetscVec*)
    int VecScatterBegin(PetscScatter,PetscVec,PetscVec,PetscInsertMode,PetscScatterMode)
    int VecScatterEnd(PetscScatter,PetscVec,PetscVec,PetscInsertMode,PetscScatterMode)

# --------------------------------------------------------------------
