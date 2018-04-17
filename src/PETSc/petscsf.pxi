# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef char* PetscSFType "const char*"
    PetscSFType PETSCSFBASIC
    PetscSFType PETSCSFWINDOW

    int PetscSFCreate(MPI_Comm,PetscSF*)
    int PetscSFSetType(PetscSF,PetscSFType)
    #int PetscSFGetType(PetscSF,PetscSFType*)
    int PetscSFSetFromOptions(PetscSF)
    int PetscSFSetUp(PetscSF)
    int PetscSFView(PetscSF,PetscViewer)
    int PetscSFReset(PetscSF)
    int PetscSFDestroy(PetscSF*)

    ctypedef struct PetscSFNode:
        PetscInt rank
        PetscInt index
    ctypedef PetscSFNode const_PetscSFNode "const PetscSFNode"
    int PetscSFGetGraph(PetscSF,PetscInt*,PetscInt*,const_PetscInt**,const_PetscSFNode**)
    int PetscSFSetGraph(PetscSF,PetscInt,PetscInt,const_PetscInt*,PetscCopyMode,PetscSFNode*,PetscCopyMode)
    int PetscSFSetRankOrder(PetscSF,PetscBool)

    int PetscSFGetMultiSF(PetscSF,PetscSF*)
    int PetscSFCreateInverseSF(PetscSF,PetscSF*)
