# --------------------------------------------------------------------

cdef extern from * nogil:

    struct _p_PetscSF
    ctypedef _p_PetscSF* PetscSF
    struct PetscSFNode:
        PetscInt rank
        PetscInt index
    ctypedef PetscSFNode PetscSFNode "PetscSFNode"
    ctypedef PetscSFNode const_PetscSFNode "const PetscSFNode"

    int PetscSFCreate(MPI_Comm,PetscSF*)
    int PetscSFSetUp(PetscSF)
    int PetscSFView(PetscSF,PetscViewer)
    int PetscSFReset(PetscSF)
    int PetscSFDestroy(PetscSF*)

    int PetscSFGetGraph(PetscSF,PetscInt*,PetscInt*,const_PetscInt**,const_PetscSFNode**)
    int PetscSFSetGraph(PetscSF,PetscInt,PetscInt,const_PetscInt*,PetscCopyMode,PetscSFNode*,PetscCopyMode)
