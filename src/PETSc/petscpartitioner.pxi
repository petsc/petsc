cdef extern from * nogil:

    ctypedef char* PetscPartitionerType "const char*"
    PetscPartitionerType PETSCPARTITIONERCHACO
    PetscPartitionerType PETSCPARTITIONERPARMETIS
    PetscPartitionerType PETSCPARTITIONERSHELL

    int PetscPartitionerCreate(MPI_Comm,PetscPartitioner*)
    int PetscPartitionerDestroy(PetscPartitioner*)
    int PetscPartitionerView(PetscPartitioner,PetscViewer)
    int PetscPartitionerSetType(PetscPartitioner,PetscPartitionerType)
    int PetscPartitionerGetType(PetscPartitioner,PetscPartitionerType*)
    int PetscPartitionerSetFromOptions(PetscPartitioner)
    int PetscPartitionerSetUp(PetscPartitioner)

    int PetscPartitionerShellSetPartition(PetscPartitioner,PetscInt,PetscInt*,PetscInt*)
