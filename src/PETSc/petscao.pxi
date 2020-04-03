cdef extern from * nogil:

    ctypedef const char* PetscAOType "AOType"
    PetscAOType AOBASIC
    PetscAOType AOADVANCED
    PetscAOType AOMAPPING
    PetscAOType AOMEMORYSCALABLE

    int AOView(PetscAO,PetscViewer)
    int AODestroy(PetscAO*)
    int AOCreateBasic(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],PetscAO*)
    int AOCreateBasicIS(PetscIS,PetscIS,PetscAO*)
    int AOCreateMemoryScalable(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],PetscAO*)
    int AOCreateMemoryScalableIS(PetscIS,PetscIS,PetscAO*)
    int AOCreateMapping(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],PetscAO*)
    int AOCreateMappingIS(PetscIS,PetscIS,PetscAO*)
    int AOGetType(PetscAO,PetscAOType*)

    int AOApplicationToPetsc(PetscAO,PetscInt,PetscInt[])
    int AOApplicationToPetscIS(PetscAO,PetscIS)
    int AOPetscToApplication(PetscAO,PetscInt,PetscInt[])
    int AOPetscToApplicationIS(PetscAO,PetscIS)
