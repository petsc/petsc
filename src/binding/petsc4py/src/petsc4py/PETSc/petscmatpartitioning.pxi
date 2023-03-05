cdef extern from * nogil:

    ctypedef const char* PetscMatPartitioningType "MatPartitioningType"
    PetscMatPartitioningType MATPARTITIONINGCURRENT
    PetscMatPartitioningType MATPARTITIONINGAVERAGE
    PetscMatPartitioningType MATPARTITIONINGSQUARE
    PetscMatPartitioningType MATPARTITIONINGPARMETIS
    PetscMatPartitioningType MATPARTITIONINGCHACO
    PetscMatPartitioningType MATPARTITIONINGPARTY
    PetscMatPartitioningType MATPARTITIONINGPTSCOTCH
    PetscMatPartitioningType MATPARTITIONINGHIERARCH

    PetscErrorCode MatPartitioningCreate(MPI_Comm,PetscMatPartitioning*)
    PetscErrorCode MatPartitioningDestroy(PetscMatPartitioning*)
    PetscErrorCode MatPartitioningView(PetscMatPartitioning,PetscViewer)

    PetscErrorCode MatPartitioningSetType(PetscMatPartitioning,PetscMatPartitioningType)
    PetscErrorCode MatPartitioningGetType(PetscMatPartitioning,PetscMatPartitioningType*)
    PetscErrorCode MatPartitioningSetFromOptions(PetscMatPartitioning)

    PetscErrorCode MatPartitioningSetAdjacency(PetscMatPartitioning,PetscMat)
    PetscErrorCode MatPartitioningApply(PetscMatPartitioning,PetscIS*)
