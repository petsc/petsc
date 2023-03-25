cdef extern from * nogil:

    ctypedef const char* PetscKSPType "KSPType"
    PetscKSPType KSPRICHARDSON
    PetscKSPType KSPCHEBYSHEV
    PetscKSPType KSPCG
    PetscKSPType KSPGROPPCG
    PetscKSPType KSPPIPECG
    PetscKSPType KSPPIPECGRR
    PetscKSPType KSPPIPELCG
    PetscKSPType KSPPIPEPRCG
    PetscKSPType KSPPIPECG2
    PetscKSPType KSPCGNE
    PetscKSPType KSPNASH
    PetscKSPType KSPSTCG
    PetscKSPType KSPGLTR
    PetscKSPType KSPFCG
    PetscKSPType KSPPIPEFCG
    PetscKSPType KSPGMRES
    PetscKSPType KSPPIPEFGMRES
    PetscKSPType   KSPFGMRES
    PetscKSPType   KSPLGMRES
    PetscKSPType   KSPDGMRES
    PetscKSPType   KSPPGMRES
    PetscKSPType KSPTCQMR
    PetscKSPType KSPBCGS
    PetscKSPType   KSPIBCGS
    PetscKSPType   KSPQMRCGS
    PetscKSPType   KSPFBCGS
    PetscKSPType   KSPFBCGSR
    PetscKSPType   KSPBCGSL
    PetscKSPType   KSPPIPEBCGS
    PetscKSPType KSPCGS
    PetscKSPType KSPTFQMR
    PetscKSPType KSPCR
    PetscKSPType KSPPIPECR
    PetscKSPType KSPLSQR
    PetscKSPType KSPPREONLY
    PetscKSPType   KSPNONE
    PetscKSPType KSPQCG
    PetscKSPType KSPBICG
    PetscKSPType KSPMINRES
    PetscKSPType KSPSYMMLQ
    PetscKSPType KSPLCD
    PetscKSPType KSPPYTHON
    PetscKSPType KSPGCR
    PetscKSPType KSPPIPEGCR
    PetscKSPType KSPTSIRM
    PetscKSPType KSPCGLS
    PetscKSPType KSPFETIDP
    PetscKSPType KSPHPDDM

    ctypedef enum PetscKSPNormType "KSPNormType":
        KSP_NORM_DEFAULT
        KSP_NORM_NONE
        KSP_NORM_PRECONDITIONED
        KSP_NORM_UNPRECONDITIONED
        KSP_NORM_NATURAL

    ctypedef enum PetscKSPConvergedReason "KSPConvergedReason":
        # iterating
        KSP_CONVERGED_ITERATING
        # converged
        KSP_CONVERGED_RTOL_NORMAL
        KSP_CONVERGED_ATOL_NORMAL
        KSP_CONVERGED_RTOL
        KSP_CONVERGED_ATOL
        KSP_CONVERGED_ITS
        KSP_CONVERGED_NEG_CURVE
        KSP_CONVERGED_STEP_LENGTH
        KSP_CONVERGED_HAPPY_BREAKDOWN
        # diverged
        KSP_DIVERGED_NULL
        KSP_DIVERGED_MAX_IT "KSP_DIVERGED_ITS"
        KSP_DIVERGED_DTOL
        KSP_DIVERGED_BREAKDOWN
        KSP_DIVERGED_BREAKDOWN_BICG
        KSP_DIVERGED_NONSYMMETRIC
        KSP_DIVERGED_INDEFINITE_PC
        KSP_DIVERGED_NANORINF
        KSP_DIVERGED_INDEFINITE_MAT
        KSP_DIVERGED_PC_FAILED

    ctypedef PetscErrorCode (*PetscKSPCtxDel)(void*)

    ctypedef PetscErrorCode (*PetscKSPConvergedFunction)(PetscKSP,
                                              PetscInt,
                                              PetscReal,
                                              PetscKSPConvergedReason*,
                                              void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscKSPMonitorFunction)(PetscKSP,
                                            PetscInt,
                                            PetscReal,
                                            void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscKSPComputeRHSFunction)(PetscKSP,
                                               PetscVec,
                                               void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscKSPComputeOpsFunction)(PetscKSP,
                                               PetscMat,
                                               PetscMat,
                                               void*) except PETSC_ERR_PYTHON

    PetscErrorCode KSPCreate(MPI_Comm,PetscKSP*)
    PetscErrorCode KSPDestroy(PetscKSP*)
    PetscErrorCode KSPView(PetscKSP,PetscViewer)

    PetscErrorCode KSPSetType(PetscKSP,PetscKSPType)
    PetscErrorCode KSPGetType(PetscKSP,PetscKSPType*)

    PetscErrorCode KSPSetOptionsPrefix(PetscKSP,char[])
    PetscErrorCode KSPAppendOptionsPrefix(PetscKSP,char[])
    PetscErrorCode KSPGetOptionsPrefix(PetscKSP,char*[])
    PetscErrorCode KSPSetFromOptions(PetscKSP)

    PetscErrorCode KSPSetTolerances(PetscKSP,PetscReal,PetscReal,PetscReal,PetscInt)
    PetscErrorCode KSPGetTolerances(PetscKSP,PetscReal*,PetscReal*,PetscReal*,PetscInt*)
    PetscErrorCode KSPSetNormType(PetscKSP,PetscKSPNormType)
    PetscErrorCode KSPGetNormType(PetscKSP,PetscKSPNormType*)
    PetscErrorCode KSPSetPCSide(PetscKSP,PetscPCSide)
    PetscErrorCode KSPGetPCSide(PetscKSP,PetscPCSide*)
    PetscErrorCode KSPSetSupportedNorm(PetscKSP,PetscKSPNormType,PetscPCSide,PetscInt)

    PetscErrorCode KSPSetConvergenceTest(PetscKSP,PetscKSPConvergedFunction,void*,PetscKSPCtxDel)
    PetscErrorCode KSPSetResidualHistory(PetscKSP,PetscReal[],PetscInt,PetscBool)
    PetscErrorCode KSPGetResidualHistory(PetscKSP,PetscReal*[],PetscInt*)
    PetscErrorCode KSPLogResidualHistory(PetscKSP,PetscReal)
    PetscErrorCode KSPConvergedDefaultCreate(void**)
    PetscErrorCode KSPConvergedDefaultDestroy(void*)
    PetscErrorCode KSPConvergedDefault(PetscKSP,PetscInt,PetscReal,PetscKSPConvergedReason*,void*) except PETSC_ERR_PYTHON
    PetscErrorCode KSPConvergedSkip(PetscKSP,PetscInt,PetscReal,PetscKSPConvergedReason*,void*) except PETSC_ERR_PYTHON

    PetscErrorCode KSPMonitorSet(PetscKSP,PetscKSPMonitorFunction,void*,PetscKSPCtxDel)
    PetscErrorCode KSPMonitorCancel(PetscKSP)
    PetscErrorCode KSPMonitor(PetscKSP,PetscInt,PetscReal)

    PetscErrorCode KSPSetInitialGuessNonzero(PetscKSP,PetscBool)
    PetscErrorCode KSPGetInitialGuessNonzero(PetscKSP,PetscBool*)
    PetscErrorCode KSPSetInitialGuessKnoll(PetscKSP,PetscBool)
    PetscErrorCode KSPGetInitialGuessKnoll(PetscKSP,PetscBool*)
    PetscErrorCode KSPSetUseFischerGuess(PetscKSP,PetscInt,PetscInt)

    PetscErrorCode KSPGetComputeEigenvalues(PetscKSP,PetscBool*)
    PetscErrorCode KSPSetComputeEigenvalues(PetscKSP,PetscBool)
    PetscErrorCode KSPGetComputeSingularValues(PetscKSP,PetscBool*)
    PetscErrorCode KSPSetComputeSingularValues(PetscKSP,PetscBool)

    PetscErrorCode KSPSetComputeRHS(PetscKSP,PetscKSPComputeRHSFunction,void*)
    PetscErrorCode KSPSetComputeOperators(PetscKSP,PetscKSPComputeOpsFunction,void*)
    PetscErrorCode KSPSetOperators(PetscKSP,PetscMat,PetscMat)
    PetscErrorCode KSPGetOperators(PetscKSP,PetscMat*,PetscMat*)
    PetscErrorCode KSPGetOperatorsSet(PetscKSP,PetscBool*,PetscBool*)

    PetscErrorCode KSPSetPC(PetscKSP,PetscPC)
    PetscErrorCode KSPGetPC(PetscKSP,PetscPC*)

    PetscErrorCode KSPGetDM(PetscKSP,PetscDM*)
    PetscErrorCode KSPSetDM(PetscKSP,PetscDM)
    PetscErrorCode KSPSetDMActive(PetscKSP,PetscBool)

    PetscErrorCode KSPSetUp(PetscKSP)
    PetscErrorCode KSPReset(PetscKSP)
    PetscErrorCode KSPSetUpOnBlocks(PetscKSP)
    PetscErrorCode KSPSolve(PetscKSP,PetscVec,PetscVec)
    PetscErrorCode KSPSolveTranspose(PetscKSP,PetscVec,PetscVec)
    PetscErrorCode KSPMatSolve(PetscKSP,PetscMat,PetscMat)
    PetscErrorCode KSPMatSolveTranspose(PetscKSP,PetscMat,PetscMat)

    PetscErrorCode KSPGetRhs(PetscKSP,PetscVec*)
    PetscErrorCode KSPGetSolution(PetscKSP,PetscVec*)
    PetscErrorCode KSPGetConvergedReason(PetscKSP,PetscKSPConvergedReason*)
    PetscErrorCode KSPGetIterationNumber(PetscKSP,PetscInt*)
    PetscErrorCode KSPGetResidualNorm(PetscKSP,PetscReal*)
    PetscErrorCode KSPSetErrorIfNotConverged(PetscKSP,PetscBool);
    PetscErrorCode KSPGetErrorIfNotConverged(PetscKSP,PetscBool*);

    PetscErrorCode KSPBuildSolution(PetscKSP,PetscVec,PetscVec*)
    PetscErrorCode KSPBuildSolutionDefault(PetscKSP,PetscVec,PetscVec*)
    PetscErrorCode KSPBuildResidual(PetscKSP,PetscVec,PetscVec,PetscVec*)
    PetscErrorCode KSPBuildResidualDefault(PetscKSP,PetscVec,PetscVec,PetscVec*)

    PetscErrorCode KSPSetDiagonalScale(PetscKSP,PetscBool)
    PetscErrorCode KSPGetDiagonalScale(PetscKSP,PetscBool*)
    PetscErrorCode KSPSetDiagonalScaleFix(PetscKSP,PetscBool)
    PetscErrorCode KSPGetDiagonalScaleFix(PetscKSP,PetscBool*)

    PetscErrorCode KSPComputeExplicitOperator(PetscKSP,PetscMat*)
    PetscErrorCode KSPComputeEigenvalues(PetscKSP,PetscInt,PetscReal[],PetscReal[],PetscInt*)
    PetscErrorCode KSPComputeExtremeSingularValues(PetscKSP,PetscReal*,PetscReal*)

    PetscErrorCode KSPCreateVecs(PetscKSP,PetscInt,PetscVec**,PetscInt,PetscVec**)

    PetscErrorCode KSPGMRESSetRestart(PetscKSP,PetscInt)

    PetscErrorCode KSPPythonSetType(PetscKSP,char[])
    PetscErrorCode KSPPythonGetType(PetscKSP,char*[])

cdef extern from * nogil: # custom.h
    PetscErrorCode KSPSetIterationNumber(PetscKSP,PetscInt)
    PetscErrorCode KSPSetResidualNorm(PetscKSP,PetscReal)
    PetscErrorCode KSPConvergenceTestCall(PetscKSP,PetscInt,PetscReal,PetscKSPConvergedReason*)
    PetscErrorCode KSPSetConvergedReason(PetscKSP,PetscKSPConvergedReason)

# -----------------------------------------------------------------------------

cdef inline KSP ref_KSP(PetscKSP ksp):
    cdef KSP ob = <KSP> KSP()
    ob.ksp = ksp
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

cdef PetscErrorCode KSP_Converged(
    PetscKSP  ksp,
    PetscInt  its,
    PetscReal rnm,
    PetscKSPConvergedReason *r,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef KSP Ksp = ref_KSP(ksp)
    (converged, args, kargs) = Ksp.get_attr('__converged__')
    reason = converged(Ksp, toInt(its), toReal(rnm), *args, **kargs)
    if   reason is None:  r[0] = KSP_CONVERGED_ITERATING
    elif reason is False: r[0] = KSP_CONVERGED_ITERATING
    elif reason is True:  r[0] = KSP_CONVERGED_ITS # XXX ?
    else:                 r[0] = reason
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode KSP_Monitor(
    PetscKSP  ksp,
    PetscInt  its,
    PetscReal rnm,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef KSP Ksp = ref_KSP(ksp)
    cdef object monitorlist = Ksp.get_attr('__monitor__')
    if monitorlist is None: return PETSC_SUCCESS
    for (monitor, args, kargs) in monitorlist:
        monitor(Ksp, toInt(its), toReal(rnm), *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode KSP_ComputeRHS(
    PetscKSP ksp,
    PetscVec rhs,
    void*    ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef KSP Ksp = ref_KSP(ksp)
    cdef Vec Rhs = ref_Vec(rhs)
    cdef object context = Ksp.get_attr('__rhs__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (computerhs, args, kargs) = context
    computerhs(Ksp, Rhs, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode KSP_ComputeOps(
    PetscKSP ksp,
    PetscMat A,
    PetscMat B,
    void*    ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef KSP Ksp  = ref_KSP(ksp)
    cdef Mat Amat = ref_Mat(A)
    cdef Mat Bmat = ref_Mat(B)
    cdef object context = Ksp.get_attr('__operators__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (computeops, args, kargs) = context
    computeops(Ksp, Amat, Bmat, *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------
