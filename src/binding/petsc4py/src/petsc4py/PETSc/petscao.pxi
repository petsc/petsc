cdef extern from * nogil:

    ctypedef const char* PetscAOType "AOType"
    PetscAOType AOBASIC
    PetscAOType AOADVANCED
    PetscAOType AOMAPPING
    PetscAOType AOMEMORYSCALABLE

    PetscErrorCode AOView(PetscAO,PetscViewer)
    PetscErrorCode AODestroy(PetscAO*)
    PetscErrorCode AOCreateBasic(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],PetscAO*)
    PetscErrorCode AOCreateBasicIS(PetscIS,PetscIS,PetscAO*)
    PetscErrorCode AOCreateMemoryScalable(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],PetscAO*)
    PetscErrorCode AOCreateMemoryScalableIS(PetscIS,PetscIS,PetscAO*)
    PetscErrorCode AOCreateMapping(MPI_Comm,PetscInt,const PetscInt[],const PetscInt[],PetscAO*)
    PetscErrorCode AOCreateMappingIS(PetscIS,PetscIS,PetscAO*)
    PetscErrorCode AOGetType(PetscAO,PetscAOType*)

    PetscErrorCode AOApplicationToPetsc(PetscAO,PetscInt,PetscInt[])
    PetscErrorCode AOApplicationToPetscIS(PetscAO,PetscIS)
    PetscErrorCode AOPetscToApplication(PetscAO,PetscInt,PetscInt[])
    PetscErrorCode AOPetscToApplicationIS(PetscAO,PetscIS)
