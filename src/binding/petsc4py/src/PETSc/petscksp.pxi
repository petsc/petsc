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
        KSP_CONVERGED_CG_NEG_CURVE
        KSP_CONVERGED_CG_CONSTRAINED
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

    ctypedef int (*PetscKSPCtxDel)(void*)

    ctypedef int (*PetscKSPConvergedFunction)(PetscKSP,
                                              PetscInt,
                                              PetscReal,
                                              PetscKSPConvergedReason*,
                                              void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscKSPMonitorFunction)(PetscKSP,
                                            PetscInt,
                                            PetscReal,
                                            void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscKSPComputeRHSFunction)(PetscKSP,
                                               PetscVec,
                                               void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscKSPComputeOpsFunction)(PetscKSP,
                                               PetscMat,
                                               PetscMat,
                                               void*) except PETSC_ERR_PYTHON

    int KSPCreate(MPI_Comm,PetscKSP*)
    int KSPDestroy(PetscKSP*)
    int KSPView(PetscKSP,PetscViewer)

    int KSPSetType(PetscKSP,PetscKSPType)
    int KSPGetType(PetscKSP,PetscKSPType*)

    int KSPSetOptionsPrefix(PetscKSP,char[])
    int KSPAppendOptionsPrefix(PetscKSP,char[])
    int KSPGetOptionsPrefix(PetscKSP,char*[])
    int KSPSetFromOptions(PetscKSP)

    int KSPSetTolerances(PetscKSP,PetscReal,PetscReal,PetscReal,PetscInt)
    int KSPGetTolerances(PetscKSP,PetscReal*,PetscReal*,PetscReal*,PetscInt*)
    int KSPSetNormType(PetscKSP,PetscKSPNormType)
    int KSPGetNormType(PetscKSP,PetscKSPNormType*)
    int KSPSetPCSide(PetscKSP,PetscPCSide)
    int KSPGetPCSide(PetscKSP,PetscPCSide*)

    int KSPSetConvergenceTest(PetscKSP,PetscKSPConvergedFunction,void*,PetscKSPCtxDel)
    int KSPSetResidualHistory(PetscKSP,PetscReal[],PetscInt,PetscBool)
    int KSPGetResidualHistory(PetscKSP,PetscReal*[],PetscInt*)
    int KSPLogResidualHistory(PetscKSP,PetscReal)
    int KSPConvergedDefaultCreate(void**)
    int KSPConvergedDefaultDestroy(void*)
    int KSPConvergedDefault(PetscKSP,PetscInt,PetscReal,PetscKSPConvergedReason*,void*) except PETSC_ERR_PYTHON
    int KSPConvergedSkip(PetscKSP,PetscInt,PetscReal,PetscKSPConvergedReason*,void*) except PETSC_ERR_PYTHON

    int KSPMonitorSet(PetscKSP,PetscKSPMonitorFunction,void*,PetscKSPCtxDel)
    int KSPMonitorCancel(PetscKSP)
    int KSPMonitor(PetscKSP,PetscInt,PetscReal)

    int KSPSetInitialGuessNonzero(PetscKSP,PetscBool)
    int KSPGetInitialGuessNonzero(PetscKSP,PetscBool*)
    int KSPSetInitialGuessKnoll(PetscKSP,PetscBool)
    int KSPGetInitialGuessKnoll(PetscKSP,PetscBool*)
    int KSPSetUseFischerGuess(PetscKSP,PetscInt,PetscInt)

    int KSPGetComputeEigenvalues(PetscKSP,PetscBool*)
    int KSPSetComputeEigenvalues(PetscKSP,PetscBool)
    int KSPGetComputeSingularValues(PetscKSP,PetscBool*)
    int KSPSetComputeSingularValues(PetscKSP,PetscBool)

    int KSPSetComputeRHS(PetscKSP,PetscKSPComputeRHSFunction,void*)
    int KSPSetComputeOperators(PetscKSP,PetscKSPComputeOpsFunction,void*)
    int KSPSetOperators(PetscKSP,PetscMat,PetscMat)
    int KSPGetOperators(PetscKSP,PetscMat*,PetscMat*)
    int KSPGetOperatorsSet(PetscKSP,PetscBool*,PetscBool*)

    int KSPSetPC(PetscKSP,PetscPC)
    int KSPGetPC(PetscKSP,PetscPC*)

    int KSPGetDM(PetscKSP,PetscDM*)
    int KSPSetDM(PetscKSP,PetscDM)
    int KSPSetDMActive(PetscKSP,PetscBool)

    int KSPSetUp(PetscKSP)
    int KSPReset(PetscKSP)
    int KSPSetUpOnBlocks(PetscKSP)
    int KSPSolve(PetscKSP,PetscVec,PetscVec)
    int KSPSolveTranspose(PetscKSP,PetscVec,PetscVec)

    int KSPGetRhs(PetscKSP,PetscVec*)
    int KSPGetSolution(PetscKSP,PetscVec*)
    int KSPGetConvergedReason(PetscKSP,PetscKSPConvergedReason*)
    int KSPGetIterationNumber(PetscKSP,PetscInt*)
    int KSPGetResidualNorm(PetscKSP,PetscReal*)
    int KSPSetErrorIfNotConverged(PetscKSP,PetscBool);
    int KSPGetErrorIfNotConverged(PetscKSP,PetscBool*);

    int KSPBuildSolution(PetscKSP,PetscVec,PetscVec*)
    int KSPBuildResidual(PetscKSP,PetscVec,PetscVec,PetscVec*)

    int KSPSetDiagonalScale(PetscKSP,PetscBool)
    int KSPGetDiagonalScale(PetscKSP,PetscBool*)
    int KSPSetDiagonalScaleFix(PetscKSP,PetscBool)
    int KSPGetDiagonalScaleFix(PetscKSP,PetscBool*)

    int KSPComputeExplicitOperator(PetscKSP,PetscMat*)
    int KSPComputeEigenvalues(PetscKSP,PetscInt,PetscReal[],PetscReal[],PetscInt*)
    int KSPComputeExtremeSingularValues(PetscKSP,PetscReal*,PetscReal*)

    int KSPCreateVecs(PetscKSP,PetscInt,PetscVec**,PetscInt,PetscVec**)

    int KSPGMRESSetRestart(PetscKSP,PetscInt)

    int KSPPythonSetType(PetscKSP,char[])
    int KSPPythonGetType(PetscKSP,char*[])

cdef extern from "custom.h" nogil:
    int KSPSetIterationNumber(PetscKSP,PetscInt)
    int KSPSetResidualNorm(PetscKSP,PetscReal)
    int KSPConvergenceTestCall(PetscKSP,PetscInt,PetscReal,PetscKSPConvergedReason*)
    int KSPSetConvergedReason(PetscKSP,PetscKSPConvergedReason)

cdef extern from "libpetsc4py.h":
    int KSPPythonSetContext(PetscKSP,void*)
    int KSPPythonGetContext(PetscKSP,void**)

# -----------------------------------------------------------------------------

cdef inline KSP ref_KSP(PetscKSP ksp):
    cdef KSP ob = <KSP> KSP()
    ob.ksp = ksp
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

cdef int KSP_Converged(
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
    return 0

# -----------------------------------------------------------------------------

cdef int KSP_Monitor(
    PetscKSP  ksp,
    PetscInt  its,
    PetscReal rnm,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef KSP Ksp = ref_KSP(ksp)
    cdef object monitorlist = Ksp.get_attr('__monitor__')
    if monitorlist is None: return 0
    for (monitor, args, kargs) in monitorlist:
        monitor(Ksp, toInt(its), toReal(rnm), *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef int KSP_ComputeRHS(
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
    return 0

cdef int KSP_ComputeOps(
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
    return 0

# -----------------------------------------------------------------------------
