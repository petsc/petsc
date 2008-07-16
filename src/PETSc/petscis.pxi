cdef extern from "petscis.h":

    ctypedef enum PetscISType "ISType":
        IS_GENERAL
        IS_STRIDE
        IS_BLOCK

    int ISView(PetscIS,PetscViewer)
    int ISDestroy(PetscIS)
    int ISCreateGeneral(MPI_Comm,PetscInt,PetscInt[],PetscIS*)
    int ISCreateBlock(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscIS*)
    int ISCreateStride(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscIS*)
    int ISGetType(PetscIS,PetscISType*)

    int ISDuplicate(PetscIS,PetscIS*)
    int ISAllGather(PetscIS,PetscIS*)
    int ISInvertPermutation(PetscIS,PetscInt,PetscIS*)

    int ISGetSize(PetscIS,PetscInt*)
    int ISGetLocalSize(PetscIS,PetscInt*)
    int ISGetIndices(PetscIS,PetscInt*[])
    int ISRestoreIndices(PetscIS,PetscInt*[])

    int ISEqual(PetscIS,PetscIS,PetscTruth*)

    int ISSetPermutation(PetscIS)
    int ISPermutation(PetscIS,PetscTruth*)
    int ISSetIdentity(PetscIS)
    int ISIdentity(PetscIS,PetscTruth*)

    int ISSort(PetscIS)
    int ISSorted(PetscIS,PetscTruth*)

    int ISSum(PetscIS,PetscIS,PetscIS*)
    int ISExpand(PetscIS,PetscIS,PetscIS*)
    int ISDifference(PetscIS,PetscIS,PetscIS*)

    int ISBlock(PetscIS,PetscTruth*)
    int ISBlockGetIndices(PetscIS,PetscInt*[])
    int ISBlockRestoreIndices(PetscIS,PetscInt*[])
    int ISBlockGetSize(PetscIS,PetscInt*)
    int ISBlockGetBlockSize(PetscIS,PetscInt*)
    int ISStride(PetscIS,PetscTruth*)
    int ISStrideGetInfo(PetscIS,PetscInt*,PetscInt*)
    int ISStrideToGeneral(PetscIS)


cdef extern from "petscis.h":

    ctypedef enum PetscGLMapType "ISGlobalToLocalMappingType":
        IS_GTOLM_MASK
        IS_GTOLM_DROP

    int ISLocalToGlobalMappingCreate(MPI_Comm,PetscInt,PetscInt[],PetscLGMap*)
    int ISLocalToGlobalMappingCreateNC(MPI_Comm,PetscInt,PetscInt[],PetscLGMap*)
    int ISLocalToGlobalMappingCreateIS(PetscIS,PetscLGMap*)
    int ISLocalToGlobalMappingBlock(PetscLGMap,PetscInt,PetscLGMap*)
    int ISLocalToGlobalMappingView(PetscLGMap,PetscViewer)
    int ISLocalToGlobalMappingDestroy(PetscLGMap)
    int ISLocalToGlobalMappingApplyIS(PetscLGMap,PetscIS,PetscIS*)
    int ISLocalToGlobalMappingGetSize(PetscLGMap,PetscInt*)
    int ISLocalToGlobalMappingGetInfo(PetscLGMap,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[])
    int ISLocalToGlobalMappingRestoreInfo(PetscLGMap,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[])
    int ISLocalToGlobalMappingBlock(PetscLGMap,PetscInt,PetscLGMap*)
    int ISLocalToGlobalMappingApply(PetscLGMap mapping,PetscInt,PetscInt[],PetscInt[])
    int ISGlobalToLocalMappingApply(PetscLGMap,PetscGLMapType,PetscInt,PetscInt[],PetscInt*,PetscInt[])
