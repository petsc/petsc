cdef extern from "petscksp.h":

    ctypedef char* PetscKSPType "const char*"
    PetscKSPType KSPRICHARDSON
    PetscKSPType KSPCHEBYCHEV
    PetscKSPType KSPCG
    PetscKSPType   KSPCGNE
    PetscKSPType   KSPNASH
    PetscKSPType   KSPSTCG
    PetscKSPType   KSPGLTR
    PetscKSPType KSPGMRES
    PetscKSPType   KSPFGMRES
    PetscKSPType   KSPLGMRES
    PetscKSPType KSPTCQMR
    PetscKSPType KSPBCGS
    PetscKSPType KSPBCGSL
    PetscKSPType KSPCGS
    PetscKSPType KSPTFQMR
    PetscKSPType KSPCR
    PetscKSPType KSPLSQR
    PetscKSPType KSPPREONLY
    PetscKSPType KSPQCG
    PetscKSPType KSPBICG
    PetscKSPType KSPMINRES
    PetscKSPType KSPSYMMLQ
    PetscKSPType KSPLCD

    ctypedef enum PetscKSPNormType "KSPNormType":
        KSP_NORM_NO
        KSP_NORM_PRECONDITIONED
        KSP_NORM_UNPRECONDITIONED
        KSP_NORM_NATURAL

    ctypedef enum PetscKSPConvergedReason "KSPConvergedReason":
        # iterating
        KSP_CONVERGED_ITERATING
        # converged
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
        KSP_DIVERGED_NAN
        KSP_DIVERGED_INDEFINITE_MAT

    ctypedef int (PetscKSPCtxDel)(void*)

    ctypedef int (PetscKSPConverged)(PetscKSP,
                                     PetscInt,
                                     PetscReal,
                                     PetscKSPConvergedReason*,
                                     void*)  except PETSC_ERR_PYTHON

    ctypedef int (PetscKSPMonitor)(PetscKSP,
                                   PetscInt,
                                   PetscReal,
                                   void*) except PETSC_ERR_PYTHON

    int KSPCreate(MPI_Comm,PetscKSP* CREATE)
    int KSPDestroy(PetscKSP)
    int KSPView(PetscKSP,PetscViewer OPTIONAL)

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
    int KSPSetPreconditionerSide(PetscKSP,PetscPCSide)
    int KSPGetPreconditionerSide(PetscKSP,PetscPCSide*)

    int KSPSetConvergenceTest(PetscKSP,PetscKSPConverged*,void*,PetscKSPCtxDel*)
    int KSPSetResidualHistory(PetscKSP,PetscReal[],PetscInt,PetscTruth)
    int KSPGetResidualHistory(PetscKSP,PetscReal*[],PetscInt*)
    int KSPDefaultConvergedCreate(void**)
    int KSPDefaultConvergedDestroy(void*)
    int KSPDefaultConverged(PetscKSP,PetscInt,PetscReal,PetscKSPConvergedReason*,void*) except PETSC_ERR_PYTHON
    int KSPSkipConverged(PetscKSP,PetscInt,PetscReal,PetscKSPConvergedReason*,void*) except PETSC_ERR_PYTHON

    int KSPMonitorSet(PetscKSP,PetscKSPMonitor*,void*,PetscKSPCtxDel*)
    int KSPMonitorCancel(PetscKSP)

    int KSPSetInitialGuessNonzero(PetscKSP,PetscTruth)
    int KSPGetInitialGuessNonzero(PetscKSP,PetscTruth*)
    int KSPSetInitialGuessKnoll(PetscKSP,PetscTruth)
    int KSPGetInitialGuessKnoll(PetscKSP,PetscTruth*)

    int KSPGetComputeEigenvalues(PetscKSP,PetscTruth*)
    int KSPSetComputeEigenvalues(PetscKSP,PetscTruth)
    int KSPGetComputeSingularValues(PetscKSP,PetscTruth*)
    int KSPSetComputeSingularValues(PetscKSP,PetscTruth)

    int KSPSetOperators(PetscKSP,PetscMat,PetscMat,PetscMatStructure)
    int KSPGetOperators(PetscKSP,PetscMat*,PetscMat*,PetscMatStructure*)
    int KSPGetOperatorsSet(PetscKSP,PetscTruth*,PetscTruth*)

    int KSPSetNullSpace(PetscKSP,PetscNullSpace)
    int KSPGetNullSpace(PetscKSP,PetscNullSpace*)

    int KSPSetPC(PetscKSP,PetscPC)
    int KSPGetPC(PetscKSP,PetscPC*)

    int KSPSetUp(PetscKSP)
    int KSPSetUpOnBlocks(PetscKSP)
    int KSPSolve(PetscKSP,PetscVec,PetscVec)
    int KSPSolveTranspose(PetscKSP,PetscVec,PetscVec)

    int KSPGetRhs(PetscKSP,PetscVec*)
    int KSPGetSolution(PetscKSP,PetscVec*)
    int KSPGetConvergedReason(PetscKSP,PetscKSPConvergedReason*)
    int KSPGetIterationNumber(PetscKSP,PetscInt*)
    int KSPGetResidualNorm(PetscKSP,PetscReal*)

    int KSPSetDiagonalScale(PetscKSP,PetscTruth)
    int KSPGetDiagonalScale(PetscKSP,PetscTruth*)
    int KSPSetDiagonalScaleFix(PetscKSP,PetscTruth)
    int KSPGetDiagonalScaleFix(PetscKSP,PetscTruth*)

    int KSPComputeExplicitOperator(PetscKSP,PetscMat*)

    int KSPGetVecs(PetscKSP,PetscInt,PetscVec**,PetscInt,PetscVec**)

cdef extern from "custom.h":
    int KSPSetIterationNumber(PetscKSP,PetscInt)
    int KSPSetResidualNorm(PetscKSP,PetscReal)
    int KSPLogResidualHistoryCall(PetscKSP,PetscInt,PetscReal)
    int KSPMonitorCall(PetscKSP,PetscInt,PetscReal)
    int KSPConvergenceTestCall(PetscKSP,PetscInt,PetscReal,PetscKSPConvergedReason*)
    int KSPSetConvergedReason(PetscKSP,PetscKSPConvergedReason)

# --------------------------------------------------------------------

cdef inline KSP ref_KSP(PetscKSP ksp):
    cdef KSP ob = <KSP> KSP()
    PetscIncref(<PetscObject>ksp)
    ob.ksp = ksp
    return ob

# --------------------------------------------------------------------

cdef inline object KSP_getCnv(PetscKSP ksp):
    return Object_getAttr(<PetscObject>ksp, "__converged__")

cdef int KSP_Converged(PetscKSP  ksp,
                       PetscInt   its,
                       PetscReal  rn,
                       PetscKSPConvergedReason *r,
                        void* ctx) except PETSC_ERR_PYTHON:
    cdef KSP Ksp = ref_KSP(ksp)
    (converged, args, kargs) = KSP_getCnv(ksp)
    reason = converged(Ksp, its, rn, *args, **kargs)
    if   reason is None:  r[0] = KSP_CONVERGED_ITERATING
    elif reason is False: r[0] = KSP_CONVERGED_ITERATING
    elif reason is True:  r[0] = KSP_CONVERGED_ITS # XXX ?
    else:                 r[0] = reason
    return 0

cdef inline int KSP_setCnvDefault(PetscKSP ksp) except -1:
    cdef void* cctx = NULL
    cdef PetscKSPNormType normtype = KSP_NORM_NO
    CHKERR( KSPGetNormType(ksp, &normtype) )
    if normtype != KSP_NORM_NO:
        CHKERR( KSPDefaultConvergedCreate(&cctx) )
        CHKERR( KSPSetConvergenceTest(
            ksp, KSPDefaultConverged,
            cctx, KSPDefaultConvergedDestroy) )
    else:
        CHKERR( KSPSetConvergenceTest(
            ksp, KSPSkipConverged,
            NULL, NULL) )
    return 0

cdef inline int KSP_setCnv(PetscKSP ksp, object cnv) except -1:
    if cnv is None: KSP_setCnvDefault(ksp)
    else: CHKERR( KSPSetConvergenceTest(ksp, KSP_Converged, NULL, NULL) )
    Object_setAttr(<PetscObject>ksp, "__converged__", cnv)
    return 0


# --------------------------------------------------------------------

cdef inline object KSP_getMon(PetscKSP ksp):
    return Object_getAttr(<PetscObject>ksp, "__monitor__")

cdef int KSP_Monitor(PetscKSP  ksp,
                     PetscInt   its,
                     PetscReal  rnorm,
                     void* ctx) except PETSC_ERR_PYTHON:
    cdef object monitorlist = KSP_getMon(ksp)
    if monitorlist is None: return 0
    cdef KSP Ksp = ref_KSP(ksp)
    for (monitor, args, kargs) in monitorlist:
        monitor(Ksp, its, rnorm, *args, **kargs)
    return 0

cdef inline int KSP_setMon(PetscKSP ksp, object mon) except -1:
    if mon is None: return 0
    CHKERR( KSPMonitorSet(ksp, KSP_Monitor, NULL, NULL) )
    cdef object monitorlist = KSP_getMon(ksp)
    if monitorlist is None: monitorlist = [mon]
    else: monitorlist.append(mon)
    Object_setAttr(<PetscObject>ksp, "__monitor__", monitorlist)
    return 0

cdef inline int KSP_delMon(PetscKSP ksp) except -1:
    CHKERR( KSPMonitorCancel(ksp) )
    Object_setAttr(<PetscObject>ksp, "__monitor__", None)
    return 0

# --------------------------------------------------------------------

cdef extern from "ctorreg.h":
    PetscKSPType KSPPYTHON
    int KSPPythonSetContext(PetscKSP,void*)
    int KSPPythonGetContext(PetscKSP,void**)

# --------------------------------------------------------------------
