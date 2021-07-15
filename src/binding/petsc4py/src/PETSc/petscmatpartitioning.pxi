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

    int MatPartitioningCreate(MPI_Comm,PetscMatPartitioning*)
    int MatPartitioningDestroy(PetscMatPartitioning*)
    int MatPartitioningView(PetscMatPartitioning,PetscViewer)

    int MatPartitioningSetType(PetscMatPartitioning,PetscMatPartitioningType)
    int MatPartitioningGetType(PetscMatPartitioning,PetscMatPartitioningType*)
    int MatPartitioningSetFromOptions(PetscMatPartitioning)

    int MatPartitioningSetAdjacency(PetscMatPartitioning,PetscMat)
    int MatPartitioningApply(PetscMatPartitioning,PetscIS*)
