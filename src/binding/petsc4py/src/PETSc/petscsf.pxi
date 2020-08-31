# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef const char* PetscSFType
    PetscSFType PETSCSFBASIC
    PetscSFType PETSCSFWINDOW

    int PetscSFCreate(MPI_Comm,PetscSF*)
    int PetscSFSetType(PetscSF,PetscSFType)
    int PetscSFGetType(PetscSF,PetscSFType*)
    int PetscSFSetFromOptions(PetscSF)
    int PetscSFSetUp(PetscSF)
    int PetscSFView(PetscSF,PetscViewer)
    int PetscSFReset(PetscSF)
    int PetscSFDestroy(PetscSF*)

    ctypedef struct PetscSFNode:
        PetscInt rank
        PetscInt index
    int PetscSFGetGraph(PetscSF,PetscInt*,PetscInt*,const PetscInt**,const PetscSFNode**)
    int PetscSFSetGraph(PetscSF,PetscInt,PetscInt,const PetscInt*,PetscCopyMode,PetscSFNode*,PetscCopyMode)
    int PetscSFSetRankOrder(PetscSF,PetscBool)

    int PetscSFComputeDegreeBegin(PetscSF,const PetscInt**)
    int PetscSFComputeDegreeEnd(PetscSF,const PetscInt**)
    int PetscSFGetMultiSF(PetscSF,PetscSF*)
    int PetscSFCreateInverseSF(PetscSF,PetscSF*)

    int PetscSFCreateEmbeddedSF(PetscSF,PetscInt,const PetscInt*,PetscSF*)
    int PetscSFCreateEmbeddedLeafSF(PetscSF,PetscInt,const PetscInt*,PetscSF*)

    int PetscSFBcastBegin(PetscSF,MPI_Datatype,const void*,void*)
    int PetscSFBcastEnd(PetscSF,MPI_Datatype,const void*,void*)
    int PetscSFReduceBegin(PetscSF,MPI_Datatype,const void*,void*,MPI_Op)
    int PetscSFReduceEnd(PetscSF,MPI_Datatype,const void*,void*,MPI_Op)
    int PetscSFScatterBegin(PetscSF,MPI_Datatype,const void*,void*)
    int PetscSFScatterEnd(PetscSF,MPI_Datatype,const void*,void*)
    int PetscSFGatherBegin(PetscSF,MPI_Datatype,const void*,void*)
    int PetscSFGatherEnd(PetscSF,MPI_Datatype,const void*,void*)
    int PetscSFFetchAndOpBegin(PetscSF,MPI_Datatype,void*,const void*,void*,MPI_Op)
    int PetscSFFetchAndOpEnd(PetscSF,MPI_Datatype,void*,const void*,void*,MPI_Op)
