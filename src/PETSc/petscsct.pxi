# --------------------------------------------------------------------

cdef extern from * nogil:

    int VecScatterView(PetscScatter,PetscViewer)
    int VecScatterDestroy(PetscScatter)
    int VecScatterCreate(PetscVec,PetscIS,PetscVec,PetscIS,PetscScatter*)
    int VecScatterCopy(PetscScatter, PetscScatter*)
    int VecScatterCreateToAll(PetscVec,PetscScatter*,PetscVec*)
    int VecScatterCreateToZero(PetscVec,PetscScatter*,PetscVec*)
    int VecScatterBegin(PetscScatter,PetscVec,PetscVec,PetscInsertMode,PetscScatterMode)
    int VecScatterEnd(PetscScatter,PetscVec,PetscVec,PetscInsertMode,PetscScatterMode)

# --------------------------------------------------------------------
