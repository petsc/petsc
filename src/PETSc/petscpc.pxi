cdef extern from * nogil:

    ctypedef char* PetscPCType "const char*"
    PetscPCType PCNONE
    PetscPCType PCJACOBI
    PetscPCType PCSOR
    PetscPCType PCLU
    PetscPCType PCSHELL
    PetscPCType PCBJACOBI
    PetscPCType PCMG
    PetscPCType PCEISENSTAT
    PetscPCType PCILU
    PetscPCType PCICC
    PetscPCType PCASM
    PetscPCType PCKSP
    PetscPCType PCCOMPOSITE
    PetscPCType PCREDUNDANT
    PetscPCType PCSPAI
    PetscPCType PCNN
    PetscPCType PCCHOLESKY
    PetscPCType PCPBJACOBI
    PetscPCType PCMAT
    PetscPCType PCHYPRE
    PetscPCType PCFIELDSPLIT
    PetscPCType PCTFS
    PetscPCType PCML
    PetscPCType PCPROMETHEUS
    PetscPCType PCGALERKIN
    PetscPCType PCEXOTIC
    PetscPCType PCHMPI
    PetscPCType PCSUPPORTGRAPH
    PetscPCType PCASA
    PetscPCType PCCP
    PetscPCType PCBFBT
    PetscPCType PCLSC
    #PetscPCType PCPYTHON
    PetscPCType PCPFMG
    PetscPCType PCSYSPFMG
    PetscPCType PCREDISTRIBUTE
    PetscPCType PCSACUSP
    PetscPCType PCSACUSPPOLY
    PetscPCType PCBICGSTABCUSP
    PetscPCType PCSVD
    PetscPCType PCAINVCUSP
    PetscPCType PCGAMG

    ctypedef enum PetscPCSide "PCSide":
        PC_LEFT
        PC_RIGHT
        PC_SYMMETRIC

    ctypedef enum PetscPCASMType "PCASMType":
        PC_ASM_BASIC
        PC_ASM_RESTRICT
        PC_ASM_INTERPOLATE
        PC_ASM_NONE

    ctypedef enum PetscPCCompositeType "PCCompositeType":
        PC_COMPOSITE_ADDITIVE
        PC_COMPOSITE_MULTIPLICATIVE
        PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE
        PC_COMPOSITE_SPECIAL
        PC_COMPOSITE_SCHUR

    int PCCreate(MPI_Comm,PetscPC*)
    int PCDestroy(PetscPC*)
    int PCView(PetscPC,PetscViewer)

    int PCSetType(PetscPC,PetscPCType)
    int PCGetType(PetscPC,PetscPCType*)

    int PCSetOptionsPrefix(PetscPC,char[])
    int PCAppendOptionsPrefix(PetscPC,char[])
    int PCGetOptionsPrefix(PetscPC,char*[])
    int PCSetFromOptions(PetscPC)

    int PCSetUp(PetscPC)
    int PCReset(PetscPC)
    int PCSetUpOnBlocks(PetscPC)

    int PCApply(PetscPC,PetscVec,PetscVec)
    int PCApplyTranspose(PetscPC,PetscVec,PetscVec)
    int PCApplySymmetricLeft(PetscPC,PetscVec,PetscVec)
    int PCApplySymmetricRight(PetscPC,PetscVec,PetscVec)
    int PCApplyRichardson(PetscPC,PetscVec,PetscVec,PetscVec,PetscReal,PetscReal,PetscReal,PetscInt)
    int PCApplyBAorAB(PetscPC,PetscPCSide,PetscVec,PetscVec,PetscVec)
    int PCApplyBAorABTranspose(PetscPC,PetscPCSide,PetscVec,PetscVec,PetscVec)

    #int PCApplyTransposeExists(PetscPC,PetscBool*)
    #int PCApplyRichardsonExists(PetscPC,PetscBool*)

    int PCGetDM(PetscPC,PetscDM*)
    int PCSetDM(PetscPC,PetscDM)

    int PCSetOperators(PetscPC,PetscMat,PetscMat,PetscMatStructure)
    int PCGetOperators(PetscPC,PetscMat*,PetscMat*,PetscMatStructure*)
    int PCGetOperatorsSet(PetscPC,PetscBool*,PetscBool*)

    int PCComputeExplicitOperator(PetscPC,PetscMat*)

    int PCDiagonalScale(PetscPC,PetscBool*)
    int PCDiagonalScaleLeft(PetscPC,PetscVec,PetscVec)
    int PCDiagonalScaleRight(PetscPC,PetscVec,PetscVec)
    int PCDiagonalScaleSet(PetscPC,PetscVec)

    int PCASMSetType(PetscPC,PetscPCASMType)
    int PCASMSetOverlap(PetscPC,PetscInt)
    int PCASMSetLocalSubdomains(PetscPC,PetscInt,PetscIS[],PetscIS[])
    int PCASMSetTotalSubdomains(PetscPC,PetscInt,PetscIS[],PetscIS[])
    int PCASMGetSubKSP(PetscPC,PetscInt*,PetscInt*,PetscKSP*[])

    int PCFactorGetMatrix(PetscPC,PetscMat*)
    int PCFactorSetZeroPivot(PetscPC,PetscReal)
    int PCFactorSetShiftType(PetscPC,PetscMatFactorShiftType)
    int PCFactorSetShiftAmount(PetscPC,PetscReal)
    int PCFactorSetMatSolverPackage(PetscPC,PetscMatSolverPackage)
    int PCFactorGetMatSolverPackage(PetscPC,PetscMatSolverPackage*)
    int PCFactorSetUpMatSolverPackage(PetscPC)
    int PCFactorSetFill(PetscPC,PetscReal)
    int PCFactorSetColumnPivot(PetscPC,PetscReal)
    int PCFactorReorderForNonzeroDiagonal(PetscPC,PetscReal)
    int PCFactorSetMatOrderingType(PetscPC,PetscMatOrderingType)
    int PCFactorSetReuseOrdering(PetscPC,PetscBool )
    int PCFactorSetReuseFill(PetscPC,PetscBool )
    int PCFactorSetUseInPlace(PetscPC)
    int PCFactorSetAllowDiagonalFill(PetscPC)
    int PCFactorSetPivotInBlocks(PetscPC,PetscBool )
    int PCFactorSetLevels(PetscPC,PetscInt)
    int PCFactorSetDropTolerance(PetscPC,PetscReal,PetscReal,PetscInt)

    int PCFieldSplitSetType(PetscPC,PetscPCCompositeType)
    int PCFieldSplitSetBlockSize(PetscPC,PetscInt)
    int PCFieldSplitSetFields(PetscPC,char[],PetscInt,PetscInt*)
    int PCFieldSplitSetIS(PetscPC,char[],PetscIS)
    int PCFieldSplitGetSubKSP(PetscPC,PetscInt*,PetscKSP*[])
    #int PCFieldSplitSchurPrecondition(PetscPC,PCFieldSplitSchurPreType,PetscMat)
    #int PCFieldSplitGetSchurBlocks(PetscPC,PetscMat*,PetscMat*,PetscMat*,PetscMat*)

# --------------------------------------------------------------------

cdef extern from "libpetsc4py.h":
    PetscPCType PCPYTHON
    int PCPythonSetContext(PetscPC,void*)
    int PCPythonGetContext(PetscPC,void**)
    int PCPythonSetType(PetscPC,char[])

# --------------------------------------------------------------------
