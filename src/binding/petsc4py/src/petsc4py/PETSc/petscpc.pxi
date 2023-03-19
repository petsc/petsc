cdef extern from * nogil:

    ctypedef const char* PetscPCType "PCType"
    PetscPCType PCNONE
    PetscPCType PCJACOBI
    PetscPCType PCSOR
    PetscPCType PCLU
    PetscPCType PCQR
    PetscPCType PCSHELL
    PetscPCType PCBJACOBI
    PetscPCType PCMG
    PetscPCType PCEISENSTAT
    PetscPCType PCILU
    PetscPCType PCICC
    PetscPCType PCASM
    PetscPCType PCGASM
    PetscPCType PCKSP
    PetscPCType PCCOMPOSITE
    PetscPCType PCREDUNDANT
    PetscPCType PCSPAI
    PetscPCType PCNN
    PetscPCType PCCHOLESKY
    PetscPCType PCPBJACOBI
    PetscPCType PCVPBJACOBI
    PetscPCType PCMAT
    PetscPCType PCHYPRE
    PetscPCType PCPARMS
    PetscPCType PCFIELDSPLIT
    PetscPCType PCTFS
    PetscPCType PCML
    PetscPCType PCGALERKIN
    PetscPCType PCEXOTIC
    PetscPCType PCCP
    PetscPCType PCBFBT
    PetscPCType PCLSC
    PetscPCType PCPYTHON
    PetscPCType PCPFMG
    PetscPCType PCSYSPFMG
    PetscPCType PCREDISTRIBUTE
    PetscPCType PCSVD
    PetscPCType PCGAMG
    PetscPCType PCCHOWILUVIENNACL
    PetscPCType PCROWSCALINGVIENNACL
    PetscPCType PCSAVIENNACL
    PetscPCType PCBDDC
    PetscPCType PCKACZMARZ
    PetscPCType PCTELESCOPE
    PetscPCType PCPATCH
    PetscPCType PCLMVM
    PetscPCType PCHMG
    PetscPCType PCDEFLATION
    PetscPCType PCHPDDM
    PetscPCType PCH2OPUS

    ctypedef enum PetscPCSide "PCSide":
        PC_SIDE_DEFAULT
        PC_LEFT
        PC_RIGHT
        PC_SYMMETRIC

    ctypedef enum PetscPCASMType "PCASMType":
        PC_ASM_BASIC
        PC_ASM_RESTRICT
        PC_ASM_INTERPOLATE
        PC_ASM_NONE

    ctypedef enum PetscPCGASMType "PCGASMType":
        PC_GASM_BASIC
        PC_GASM_RESTRICT
        PC_GASM_INTERPOLATE
        PC_GASM_NONE

    ctypedef enum PetscPCMGType "PCMGType":
        PC_MG_MULTIPLICATIVE
        PC_MG_ADDITIVE
        PC_MG_FULL
        PC_MG_KASKADE

    ctypedef enum PetscPCMGCycleType "PCMGCycleType":
        PC_MG_CYCLE_V
        PC_MG_CYCLE_W

    ctypedef const char* PetscPCGAMGType "PCGAMGType"
    PetscPCGAMGType PCGAMGAGG
    PetscPCGAMGType PCGAMGGEO
    PetscPCGAMGType PCGAMGCLASSICAL

    ctypedef const char* PetscPCHYPREType "const char*"

    ctypedef enum PetscPCCompositeType "PCCompositeType":
        PC_COMPOSITE_ADDITIVE
        PC_COMPOSITE_MULTIPLICATIVE
        PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE
        PC_COMPOSITE_SPECIAL
        PC_COMPOSITE_SCHUR

    ctypedef enum PetscPCFieldSplitSchurPreType "PCFieldSplitSchurPreType":
        PC_FIELDSPLIT_SCHUR_PRE_SELF
        PC_FIELDSPLIT_SCHUR_PRE_SELFP
        PC_FIELDSPLIT_SCHUR_PRE_A11
        PC_FIELDSPLIT_SCHUR_PRE_USER
        PC_FIELDSPLIT_SCHUR_PRE_FULL

    ctypedef enum PetscPCFieldSplitSchurFactType "PCFieldSplitSchurFactType":
        PC_FIELDSPLIT_SCHUR_FACT_DIAG
        PC_FIELDSPLIT_SCHUR_FACT_LOWER
        PC_FIELDSPLIT_SCHUR_FACT_UPPER
        PC_FIELDSPLIT_SCHUR_FACT_FULL

    ctypedef enum PetscPCPatchConstructType "PCPatchConstructType":
        PC_PATCH_STAR
        PC_PATCH_VANKA
        PC_PATCH_PARDECOMP
        PC_PATCH_USER
        PC_PATCH_PYTHON

    ctypedef enum PetscPCHPDDMCoarseCorrectionType "PCHPDDMCoarseCorrectionType":
        PC_HPDDM_COARSE_CORRECTION_DEFLATED
        PC_HPDDM_COARSE_CORRECTION_ADDITIVE
        PC_HPDDM_COARSE_CORRECTION_BALANCED

    ctypedef enum PetscPCDeflationSpaceType "PCDeflationSpaceType":
        PC_DEFLATION_SPACE_HAAR
        PC_DEFLATION_SPACE_DB2
        PC_DEFLATION_SPACE_DB4
        PC_DEFLATION_SPACE_DB8
        PC_DEFLATION_SPACE_DB16
        PC_DEFLATION_SPACE_BIORTH22
        PC_DEFLATION_SPACE_MEYER
        PC_DEFLATION_SPACE_AGGREGATION
        PC_DEFLATION_SPACE_USER

    ctypedef enum PetscPCFailedReason "PCFailedReason":
        PC_SETUP_ERROR
        PC_NOERROR
        PC_FACTOR_STRUCT_ZEROPIVOT
        PC_FACTOR_NUMERIC_ZEROPIVOT
        PC_FACTOR_OUTMEMORY
        PC_FACTOR_OTHER
        PC_SUBPC_ERROR

    PetscErrorCode PCCreate(MPI_Comm,PetscPC*)
    PetscErrorCode PCDestroy(PetscPC*)
    PetscErrorCode PCView(PetscPC,PetscViewer)

    PetscErrorCode PCSetType(PetscPC,PetscPCType)
    PetscErrorCode PCGetType(PetscPC,PetscPCType*)

    PetscErrorCode PCSetOptionsPrefix(PetscPC,char[])
    PetscErrorCode PCAppendOptionsPrefix(PetscPC,char[])
    PetscErrorCode PCGetOptionsPrefix(PetscPC,char*[])
    PetscErrorCode PCSetFromOptions(PetscPC)

    PetscErrorCode PCSetFailedReason(PetscPC,PetscPCFailedReason)
    PetscErrorCode PCGetFailedReason(PetscPC,PetscPCFailedReason*)
    PetscErrorCode PCGetFailedReasonRank(PetscPC,PetscPCFailedReason*)

    PetscErrorCode PCSetUp(PetscPC)
    PetscErrorCode PCReset(PetscPC)
    PetscErrorCode PCSetUpOnBlocks(PetscPC)

    PetscErrorCode PCApply(PetscPC,PetscVec,PetscVec)
    PetscErrorCode PCMatApply(PetscPC,PetscMat,PetscMat)
    PetscErrorCode PCApplyTranspose(PetscPC,PetscVec,PetscVec)
    PetscErrorCode PCApplySymmetricLeft(PetscPC,PetscVec,PetscVec)
    PetscErrorCode PCApplySymmetricRight(PetscPC,PetscVec,PetscVec)
    PetscErrorCode PCApplyRichardson(PetscPC,PetscVec,PetscVec,PetscVec,PetscReal,PetscReal,PetscReal,PetscInt)
    PetscErrorCode PCApplyBAorAB(PetscPC,PetscPCSide,PetscVec,PetscVec,PetscVec)
    PetscErrorCode PCApplyBAorABTranspose(PetscPC,PetscPCSide,PetscVec,PetscVec,PetscVec)

    #int PCApplyTransposeExists(PetscPC,PetscBool*)
    #int PCApplyRichardsonExists(PetscPC,PetscBool*)

    PetscErrorCode PCGetDM(PetscPC,PetscDM*)
    PetscErrorCode PCSetDM(PetscPC,PetscDM)

    PetscErrorCode PCSetOperators(PetscPC,PetscMat,PetscMat)
    PetscErrorCode PCGetOperators(PetscPC,PetscMat*,PetscMat*)
    PetscErrorCode PCGetOperatorsSet(PetscPC,PetscBool*,PetscBool*)
    PetscErrorCode PCSetCoordinates(PetscPC,PetscInt,PetscInt,PetscReal[])
    PetscErrorCode PCSetUseAmat(PetscPC,PetscBool)
    PetscErrorCode PCGetUseAmat(PetscPC,PetscBool*)

    PetscErrorCode PCComputeExplicitOperator(PetscPC,PetscMat*)

    PetscErrorCode PCDiagonalScale(PetscPC,PetscBool*)
    PetscErrorCode PCDiagonalScaleLeft(PetscPC,PetscVec,PetscVec)
    PetscErrorCode PCDiagonalScaleRight(PetscPC,PetscVec,PetscVec)
    PetscErrorCode PCDiagonalScaleSet(PetscPC,PetscVec)

    PetscErrorCode PCASMSetType(PetscPC,PetscPCASMType)
    PetscErrorCode PCASMSetOverlap(PetscPC,PetscInt)
    PetscErrorCode PCASMSetLocalSubdomains(PetscPC,PetscInt,PetscIS[],PetscIS[])
    PetscErrorCode PCASMSetTotalSubdomains(PetscPC,PetscInt,PetscIS[],PetscIS[])
    PetscErrorCode PCASMGetSubKSP(PetscPC,PetscInt*,PetscInt*,PetscKSP*[])
    PetscErrorCode PCASMSetSortIndices(PetscPC,PetscBool)

    PetscErrorCode PCGASMSetType(PetscPC,PetscPCGASMType)
    PetscErrorCode PCGASMSetOverlap(PetscPC,PetscInt)

    PetscErrorCode PCGAMGSetType(PetscPC,PetscPCGAMGType)
    PetscErrorCode PCGAMGSetNlevels(PetscPC,PetscInt)
    PetscErrorCode PCGAMGSetNSmooths(PetscPC,PetscInt)

    PetscErrorCode PCHYPREGetType(PetscPC,PetscPCHYPREType*)
    PetscErrorCode PCHYPRESetType(PetscPC,PetscPCHYPREType)
    PetscErrorCode PCHYPRESetDiscreteCurl(PetscPC,PetscMat);
    PetscErrorCode PCHYPRESetDiscreteGradient(PetscPC,PetscMat);
    PetscErrorCode PCHYPRESetAlphaPoissonMatrix(PetscPC,PetscMat);
    PetscErrorCode PCHYPRESetBetaPoissonMatrix(PetscPC,PetscMat);
    PetscErrorCode PCHYPRESetEdgeConstantVectors(PetscPC,PetscVec,PetscVec,PetscVec);
    PetscErrorCode PCHYPRESetInterpolations(PetscPC, PetscInt, PetscMat, PetscMat[], PetscMat, PetscMat[]);
    PetscErrorCode PCHYPREAMSSetInteriorNodes(PetscPC, PetscVec);

    PetscErrorCode PCFactorGetMatrix(PetscPC,PetscMat*)
    PetscErrorCode PCFactorSetZeroPivot(PetscPC,PetscReal)
    PetscErrorCode PCFactorSetShiftType(PetscPC,PetscMatFactorShiftType)
    PetscErrorCode PCFactorSetShiftAmount(PetscPC,PetscReal)
    PetscErrorCode PCFactorSetMatSolverType(PetscPC,PetscMatSolverType)
    PetscErrorCode PCFactorGetMatSolverType(PetscPC,PetscMatSolverType*)
    PetscErrorCode PCFactorSetUpMatSolverType(PetscPC)
    PetscErrorCode PCFactorSetFill(PetscPC,PetscReal)
    PetscErrorCode PCFactorSetColumnPivot(PetscPC,PetscReal)
    PetscErrorCode PCFactorReorderForNonzeroDiagonal(PetscPC,PetscReal)
    PetscErrorCode PCFactorSetMatOrderingType(PetscPC,PetscMatOrderingType)
    PetscErrorCode PCFactorSetReuseOrdering(PetscPC,PetscBool )
    PetscErrorCode PCFactorSetReuseFill(PetscPC,PetscBool )
    PetscErrorCode PCFactorSetUseInPlace(PetscPC)
    PetscErrorCode PCFactorSetAllowDiagonalFill(PetscPC)
    PetscErrorCode PCFactorSetPivotInBlocks(PetscPC,PetscBool )
    PetscErrorCode PCFactorSetLevels(PetscPC,PetscInt)
    PetscErrorCode PCFactorSetDropTolerance(PetscPC,PetscReal,PetscReal,PetscInt)

    PetscErrorCode PCFieldSplitSetType(PetscPC,PetscPCCompositeType)
    PetscErrorCode PCFieldSplitSetBlockSize(PetscPC,PetscInt)
    PetscErrorCode PCFieldSplitSetFields(PetscPC,char[],PetscInt,PetscInt*,PetscInt*)
    PetscErrorCode PCFieldSplitSetIS(PetscPC,char[],PetscIS)
    PetscErrorCode PCFieldSplitGetSubKSP(PetscPC,PetscInt*,PetscKSP*[])
    PetscErrorCode PCFieldSplitSchurGetSubKSP(PetscPC,PetscInt*,PetscKSP*[])
    PetscErrorCode PCFieldSplitSetSchurPre(PetscPC,PetscPCFieldSplitSchurPreType,PetscMat)
    PetscErrorCode PCFieldSplitSetSchurFactType(PetscPC,PetscPCFieldSplitSchurFactType)
    #int PCFieldSplitGetSchurBlocks(PetscPC,PetscMat*,PetscMat*,PetscMat*,PetscMat*)

    PetscErrorCode PCCompositeSetType(PetscPC,PetscPCCompositeType)
    PetscErrorCode PCCompositeGetPC(PetscPC,PetscInt,PetscPC*)
    PetscErrorCode PCCompositeAddPCType(PetscPC,PetscPCType)
    PetscErrorCode PCCompositeAddPC(PetscPC,PetscPC)

    PetscErrorCode PCKSPGetKSP(PetscPC,PetscKSP*)

    PetscErrorCode PCSetReusePreconditioner(PetscPC,PetscBool)

    # --- MG ---
    PetscErrorCode PCMGSetType(PetscPC,PetscPCMGType)
    PetscErrorCode PCMGGetType(PetscPC,PetscPCMGType*)
    PetscErrorCode PCMGSetInterpolation(PetscPC,PetscInt,PetscMat)
    PetscErrorCode PCMGGetInterpolation(PetscPC,PetscInt,PetscMat*)
    PetscErrorCode PCMGSetRestriction(PetscPC,PetscInt,PetscMat)
    PetscErrorCode PCMGGetRestriction(PetscPC,PetscInt,PetscMat*)
    PetscErrorCode PCMGSetRScale(PetscPC,PetscInt,PetscVec)
    PetscErrorCode PCMGGetRScale(PetscPC,PetscInt,PetscVec*)
    PetscErrorCode PCMGGetSmoother(PetscPC,PetscInt,PetscKSP*)
    PetscErrorCode PCMGGetSmootherUp(PetscPC,PetscInt,PetscKSP*)
    PetscErrorCode PCMGGetSmootherDown(PetscPC,PetscInt,PetscKSP*)
    PetscErrorCode PCMGGetCoarseSolve(PetscPC,PetscKSP*)
    PetscErrorCode PCMGSetRhs(PetscPC,PetscInt,PetscVec)
    PetscErrorCode PCMGSetX(PetscPC,PetscInt,PetscVec)
    PetscErrorCode PCMGSetR(PetscPC,PetscInt,PetscVec)
    PetscErrorCode PCMGSetLevels(PetscPC,PetscInt,MPI_Comm*)
    PetscErrorCode PCMGGetLevels(PetscPC,PetscInt*)
    PetscErrorCode PCMGSetCycleType(PetscPC,PetscPCMGCycleType)
    PetscErrorCode PCMGSetCycleTypeOnLevel(PetscPC,PetscInt,PetscPCMGCycleType)
    PetscErrorCode PCBDDCSetDiscreteGradient(PetscPC,PetscMat,PetscInt,PetscInt,PetscBool,PetscBool)
    PetscErrorCode PCBDDCSetDivergenceMat(PetscPC,PetscMat,PetscBool,PetscIS)
    PetscErrorCode PCBDDCSetChangeOfBasisMat(PetscPC,PetscMat,PetscBool)
    PetscErrorCode PCBDDCSetPrimalVerticesIS(PetscPC,PetscIS)
    PetscErrorCode PCBDDCSetPrimalVerticesLocalIS(PetscPC,PetscIS)
    PetscErrorCode PCBDDCSetCoarseningRatio(PetscPC,PetscInt)
    PetscErrorCode PCBDDCSetLevels(PetscPC,PetscInt)
    PetscErrorCode PCBDDCSetDirichletBoundaries(PetscPC,PetscIS)
    PetscErrorCode PCBDDCSetDirichletBoundariesLocal(PetscPC,PetscIS)
    PetscErrorCode PCBDDCSetNeumannBoundaries(PetscPC,PetscIS)
    PetscErrorCode PCBDDCSetNeumannBoundariesLocal(PetscPC,PetscIS)
    PetscErrorCode PCBDDCSetDofsSplitting(PetscPC,PetscInt,PetscIS[])
    PetscErrorCode PCBDDCSetDofsSplittingLocal(PetscPC,PetscInt,PetscIS[])

    # --- Patch ---
    ctypedef PetscErrorCode (*PetscPCPatchComputeOperator)(PetscPC,
                                                PetscInt,
                                                PetscVec,
                                                PetscMat,
                                                PetscIS,
                                                PetscInt,
                                                const PetscInt*,
                                                const PetscInt*,
                                                void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscPCPatchComputeFunction)(PetscPC,
                                                PetscInt,
                                                PetscVec,
                                                PetscVec,
                                                PetscIS,
                                                PetscInt,
                                                const PetscInt*,
                                                const PetscInt*,
                                                void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscPCPatchConstructOperator)(PetscPC,
                                                  PetscInt*,
                                                  PetscIS**,
                                                  PetscIS*,
                                                  void*) except PETSC_ERR_PYTHON
    PetscErrorCode PCPatchSetCellNumbering(PetscPC, PetscSection)
    PetscErrorCode PCPatchSetDiscretisationInfo(PetscPC, PetscInt, PetscDM*, PetscInt*, PetscInt*, const PetscInt**, const PetscInt*, PetscInt, const PetscInt*, PetscInt, const PetscInt*)
    PetscErrorCode PCPatchSetComputeOperator(PetscPC, PetscPCPatchComputeOperator, void*)
    PetscErrorCode PCPatchSetComputeOperatorInteriorFacets(PetscPC, PetscPCPatchComputeOperator, void*)
    PetscErrorCode PCPatchSetComputeFunction(PetscPC, PetscPCPatchComputeFunction, void*)
    PetscErrorCode PCPatchSetComputeFunctionInteriorFacets(PetscPC, PetscPCPatchComputeFunction, void*)
    PetscErrorCode PCPatchSetConstructType(PetscPC, PetscPCPatchConstructType, PetscPCPatchConstructOperator, void*)

    ctypedef PetscErrorCode (*PetscPCHPDDMAuxiliaryMat)(PetscMat,
                                             PetscReal,
                                             PetscVec,
                                             PetscVec,
                                             PetscReal,
                                             PetscIS,
                                             void*) except PETSC_ERR_PYTHON
    PetscErrorCode PCHPDDMSetAuxiliaryMat(PetscPC,PetscIS,PetscMat,PetscPCHPDDMAuxiliaryMat,void*)
    PetscErrorCode PCHPDDMSetRHSMat(PetscPC,PetscMat)
    PetscErrorCode PCHPDDMHasNeumannMat(PetscPC,PetscBool)
    PetscErrorCode PCHPDDMSetCoarseCorrectionType(PetscPC,PetscPCHPDDMCoarseCorrectionType)
    PetscErrorCode PCHPDDMGetCoarseCorrectionType(PetscPC,PetscPCHPDDMCoarseCorrectionType*)
    PetscErrorCode PCHPDDMGetSTShareSubKSP(PetscPC,PetscBool*)
    PetscErrorCode PCHPDDMSetDeflationMat(PetscPC,PetscIS,PetscMat)

    # --- SPAI ---
    PetscErrorCode PCSPAISetEpsilon(PetscPC,PetscReal)
    PetscErrorCode PCSPAISetNBSteps(PetscPC,PetscInt)
    PetscErrorCode PCSPAISetMax(PetscPC,PetscInt)
    PetscErrorCode PCSPAISetMaxNew(PetscPC,PetscInt)
    PetscErrorCode PCSPAISetBlockSize(PetscPC,PetscInt)
    PetscErrorCode PCSPAISetCacheSize(PetscPC,PetscInt)
    PetscErrorCode PCSPAISetVerbose(PetscPC,PetscInt)
    PetscErrorCode PCSPAISetSp(PetscPC,PetscInt)

    # --- DEFLATION ---
    PetscErrorCode PCDeflationSetInitOnly(PetscPC,PetscBool)
    PetscErrorCode PCDeflationSetLevels(PetscPC,PetscInt)
    PetscErrorCode PCDeflationSetReductionFactor(PetscPC,PetscInt)
    PetscErrorCode PCDeflationSetCorrectionFactor(PetscPC,PetscScalar)
    PetscErrorCode PCDeflationSetSpaceToCompute(PetscPC,PetscPCDeflationSpaceType,PetscInt)
    PetscErrorCode PCDeflationSetSpace(PetscPC,PetscMat,PetscBool)
    PetscErrorCode PCDeflationSetProjectionNullSpaceMat(PetscPC,PetscMat)
    PetscErrorCode PCDeflationSetCoarseMat(PetscPC,PetscMat)
    PetscErrorCode PCDeflationGetCoarseKSP(PetscPC,PetscKSP*)
    PetscErrorCode PCDeflationGetPC(PetscPC,PetscPC*)

    # --- PYTHON ---
    PetscErrorCode PCPythonSetType(PetscPC,char[])
    PetscErrorCode PCPythonGetType(PetscPC,char*[])

# --------------------------------------------------------------------

cdef inline PC ref_PC(PetscPC pc):
    cdef PC ob = <PC> PC()
    ob.pc = pc
    PetscINCREF(ob.obj)
    return ob

cdef PetscErrorCode PCPatch_ComputeOperator(
    PetscPC pc,
    PetscInt point,
    PetscVec vec,
    PetscMat mat,
    PetscIS cells,
    PetscInt ndof,
    const PetscInt *dofmap,
    const PetscInt *dofmapWithAll,
    void *ctx) except PETSC_ERR_PYTHON with gil:
    cdef Vec Vec = ref_Vec(vec)
    cdef Mat Mat = ref_Mat(mat)
    cdef PC Pc = ref_PC(pc)
    cdef IS Is = ref_IS(cells)
    cdef object context = Pc.get_attr("__patch_compute_operator__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple
    (op, args, kargs) = context
    cdef PetscInt[:] pydofs = <PetscInt[:ndof]>dofmap
    cdef PetscInt[:] pydofsWithAll
    if dofmapWithAll != NULL:
        pydofsWithAll = <PetscInt[:ndof]>dofmapWithAll
        dofsall = asarray(pydofsWithAll)
    else:
        dofsall = None
    op(Pc, toInt(point), Vec, Mat, Is, asarray(pydofs), dofsall, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode PCPatch_ComputeFunction(
    PetscPC pc,
    PetscInt point,
    PetscVec vec,
    PetscVec out,
    PetscIS cells,
    PetscInt ndof,
    const PetscInt *dofmap,
    const PetscInt *dofmapWithAll,
    void *ctx) except PETSC_ERR_PYTHON with gil:
    cdef Vec Out = ref_Vec(out)
    cdef Vec Vec = ref_Vec(vec)
    cdef PC Pc = ref_PC(pc)
    cdef IS Is = ref_IS(cells)
    cdef object context = Pc.get_attr("__patch_compute_function__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple
    (op, args, kargs) = context
    cdef PetscInt[:] pydofs = <PetscInt[:ndof]>dofmap
    cdef PetscInt[:] pydofsWithAll = <PetscInt[:ndof]>dofmapWithAll
    op(Pc, toInt(point), Vec, Out, Is, asarray(pydofs), asarray(pydofsWithAll), *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode PCPatch_ComputeOperatorInteriorFacets(
    PetscPC pc,
    PetscInt point,
    PetscVec vec,
    PetscMat mat,
    PetscIS facets,
    PetscInt ndof,
    const PetscInt *dofmap,
    const PetscInt *dofmapWithAll,
    void *ctx) except PETSC_ERR_PYTHON with gil:
    cdef Vec Vec = ref_Vec(vec)
    cdef Mat Mat = ref_Mat(mat)
    cdef PC Pc = ref_PC(pc)
    cdef IS Is = ref_IS(facets)
    cdef object context = Pc.get_attr("__patch_compute_operator_interior_facets__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple
    (op, args, kargs) = context
    cdef PetscInt[:] pydofs = <PetscInt[:ndof]>dofmap
    cdef PetscInt[:] pydofsWithAll
    if dofmapWithAll != NULL:
        pydofsWithAll = <PetscInt[:ndof]>dofmapWithAll
        dofsall = asarray(pydofsWithAll)
    else:
        dofsall = None
    op(Pc, toInt(point), Vec, Mat, Is, asarray(pydofs), dofsall, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode PCPatch_ComputeFunctionInteriorFacets(
    PetscPC pc,
    PetscInt point,
    PetscVec vec,
    PetscVec out,
    PetscIS facets,
    PetscInt ndof,
    const PetscInt *dofmap,
    const PetscInt *dofmapWithAll,
    void *ctx) except PETSC_ERR_PYTHON with gil:
    cdef Vec Out = ref_Vec(out)
    cdef Vec Vec = ref_Vec(vec)
    cdef PC Pc = ref_PC(pc)
    cdef IS Is = ref_IS(facets)
    cdef object context = Pc.get_attr("__patch_compute_function_interior_facets__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple
    (op, args, kargs) = context
    cdef PetscInt[:] pydofs = <PetscInt[:ndof]>dofmap
    cdef PetscInt[:] pydofsWithAll = <PetscInt[:ndof]>dofmapWithAll
    op(Pc, toInt(point), Vec, Out, Is, asarray(pydofs), asarray(pydofsWithAll), *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode PCPatch_UserConstructOperator(
    PetscPC pc,
    PetscInt *n,
    PetscIS **userIS,
    PetscIS *userIterationSet,
    void *ctx) except PETSC_ERR_PYTHON with gil:
    cdef PC Pc = ref_PC(pc)
    cdef PetscInt i
    cdef object context = Pc.get_attr("__patch_construction_operator__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple
    (op, args, kargs) = context
    (patches, iterationSet) = op(Pc, *args, **kargs)
    n[0] = len(patches)
    CHKERR(PetscMalloc(<size_t>n[0]*sizeof(PetscIS), userIS))
    for i in range(n[0]):
        userIS[0][i] = (<IS?>patches[i]).iset
        PetscINCREF(<PetscObject*>&(userIS[0][i]))
    userIterationSet[0] = (<IS?>iterationSet).iset
    PetscINCREF(<PetscObject*>&(userIterationSet[0]))
    return PETSC_SUCCESS
