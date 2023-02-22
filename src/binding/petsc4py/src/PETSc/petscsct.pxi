# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef PetscSFType PetscScatterType  "VecScatterType"

    PetscErrorCode VecScatterView(PetscScatter,PetscViewer)
    PetscErrorCode VecScatterDestroy(PetscScatter*)
    PetscErrorCode VecScatterSetUp(PetscScatter)
    PetscErrorCode VecScatterCreate(PetscVec,PetscIS,PetscVec,PetscIS,PetscScatter*)
    PetscErrorCode VecScatterSetFromOptions(PetscScatter)
    PetscErrorCode VecScatterSetType(PetscScatter,PetscScatterType)
    PetscErrorCode VecScatterGetType(PetscScatter,PetscScatterType*)
    PetscErrorCode VecScatterCopy(PetscScatter, PetscScatter*)
    PetscErrorCode VecScatterCreateToAll(PetscVec,PetscScatter*,PetscVec*)
    PetscErrorCode VecScatterCreateToZero(PetscVec,PetscScatter*,PetscVec*)
    PetscErrorCode VecScatterBegin(PetscScatter,PetscVec,PetscVec,PetscInsertMode,PetscScatterMode)
    PetscErrorCode VecScatterEnd(PetscScatter,PetscVec,PetscVec,PetscInsertMode,PetscScatterMode)

# --------------------------------------------------------------------
