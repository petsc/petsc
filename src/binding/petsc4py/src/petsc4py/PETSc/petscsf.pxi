# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef const char* PetscSFType
    PetscSFType PETSCSFBASIC
    PetscSFType PETSCSFNEIGHBOR
    PetscSFType PETSCSFALLGATHERV
    PetscSFType PETSCSFALLGATHER
    PetscSFType PETSCSFGATHERV
    PetscSFType PETSCSFGATHER
    PetscSFType PETSCSFALLTOALL
    PetscSFType PETSCSFWINDOW

    PetscErrorCode PetscSFCreate(MPI_Comm,PetscSF*)
    PetscErrorCode PetscSFSetType(PetscSF,PetscSFType)
    PetscErrorCode PetscSFGetType(PetscSF,PetscSFType*)
    PetscErrorCode PetscSFSetFromOptions(PetscSF)
    PetscErrorCode PetscSFSetUp(PetscSF)
    PetscErrorCode PetscSFView(PetscSF,PetscViewer)
    PetscErrorCode PetscSFReset(PetscSF)
    PetscErrorCode PetscSFDestroy(PetscSF*)

    ctypedef struct PetscSFNode:
        PetscInt rank
        PetscInt index
    PetscErrorCode PetscSFGetGraph(PetscSF,PetscInt*,PetscInt*,const PetscInt**,const PetscSFNode**)
    PetscErrorCode PetscSFSetGraph(PetscSF,PetscInt,PetscInt,const PetscInt*,PetscCopyMode,PetscSFNode*,PetscCopyMode)
    PetscErrorCode PetscSFSetRankOrder(PetscSF,PetscBool)

    PetscErrorCode PetscSFComputeDegreeBegin(PetscSF,const PetscInt**)
    PetscErrorCode PetscSFComputeDegreeEnd(PetscSF,const PetscInt**)
    PetscErrorCode PetscSFGetMultiSF(PetscSF,PetscSF*)
    PetscErrorCode PetscSFCreateInverseSF(PetscSF,PetscSF*)

    PetscErrorCode PetscSFCreateEmbeddedRootSF(PetscSF,PetscInt,const PetscInt*,PetscSF*)
    PetscErrorCode PetscSFCreateEmbeddedLeafSF(PetscSF,PetscInt,const PetscInt*,PetscSF*)

    PetscErrorCode PetscSFDistributeSection(PetscSF,PetscSection,PetscInt**,PetscSection)
    PetscErrorCode PetscSFCreateSectionSF(PetscSF,PetscSection,PetscInt*,PetscSection, PetscSF*)

    PetscErrorCode PetscSFCompose(PetscSF,PetscSF,PetscSF*)

    PetscErrorCode PetscSFBcastBegin(PetscSF,MPI_Datatype,const void*,void*,MPI_Op)
    PetscErrorCode PetscSFBcastEnd(PetscSF,MPI_Datatype,const void*,void*,MPI_Op)
    PetscErrorCode PetscSFReduceBegin(PetscSF,MPI_Datatype,const void*,void*,MPI_Op)
    PetscErrorCode PetscSFReduceEnd(PetscSF,MPI_Datatype,const void*,void*,MPI_Op)
    PetscErrorCode PetscSFScatterBegin(PetscSF,MPI_Datatype,const void*,void*)
    PetscErrorCode PetscSFScatterEnd(PetscSF,MPI_Datatype,const void*,void*)
    PetscErrorCode PetscSFGatherBegin(PetscSF,MPI_Datatype,const void*,void*)
    PetscErrorCode PetscSFGatherEnd(PetscSF,MPI_Datatype,const void*,void*)
    PetscErrorCode PetscSFFetchAndOpBegin(PetscSF,MPI_Datatype,void*,const void*,void*,MPI_Op)
    PetscErrorCode PetscSFFetchAndOpEnd(PetscSF,MPI_Datatype,void*,const void*,void*,MPI_Op)
