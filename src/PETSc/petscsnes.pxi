cdef extern from "petscsnes.h" nogil:

    ctypedef char* PetscSNESType "const char*"
    PetscSNESType SNESLS
    PetscSNESType SNESTR
    PetscSNESType SNESTEST

    ctypedef enum PetscSNESConvergedReason "SNESConvergedReason":
      # iterating
      SNES_CONVERGED_ITERATING
      # converged
      SNES_CONVERGED_FNORM_ABS
      SNES_CONVERGED_FNORM_RELATIVE
      SNES_CONVERGED_PNORM_RELATIVE
      SNES_CONVERGED_ITS
      SNES_CONVERGED_TR_DELTA
      # diverged
      SNES_DIVERGED_FUNCTION_DOMAIN
      SNES_DIVERGED_FUNCTION_COUNT
      SNES_DIVERGED_LINEAR_SOLVE
      SNES_DIVERGED_FNORM_NAN
      SNES_DIVERGED_MAX_IT
      SNES_DIVERGED_LS_FAILURE
      SNES_DIVERGED_LOCAL_MIN


    ctypedef int PetscSNESCtxDel(void*)

    ctypedef int PetscSNESFunction(PetscSNES,
                                   PetscVec,
                                   PetscVec,
                                   void*) except PETSC_ERR_PYTHON

    ctypedef int PetscSNESUpdate(PetscSNES,
                                 PetscInt) except PETSC_ERR_PYTHON

    ctypedef int PetscSNESJacobian(PetscSNES,
                                   PetscVec,
                                   PetscMat*,
                                   PetscMat*,
                                   PetscMatStructure*,
                                   void*) except PETSC_ERR_PYTHON

    ctypedef int PetscSNESConverged(PetscSNES,
                                    PetscInt,
                                    PetscReal,
                                    PetscReal,
                                    PetscReal,
                                    PetscSNESConvergedReason*,
                                    void*)  except PETSC_ERR_PYTHON

    ctypedef int PetscSNESMonitor(PetscSNES,
                                  PetscInt,
                                  PetscReal,
                                  void*) except PETSC_ERR_PYTHON

    int SNESCreate(MPI_Comm,PetscSNES*)
    int SNESDestroy(PetscSNES)
    int SNESView(PetscSNES,PetscViewer)

    int SNESSetType(PetscSNES,PetscSNESType)
    int SNESGetType(PetscSNES,PetscSNESType*)
    int SNESSetOptionsPrefix(PetscSNES,char[])
    int SNESAppendOptionsPrefix(PetscSNES,char[])
    int SNESGetOptionsPrefix(PetscSNES,char*[])
    int SNESSetFromOptions(PetscSNES)

    int SNESGetKSP(PetscSNES,PetscKSP*)
    int SNESSetKSP(PetscSNES,PetscKSP)

    int SNESGetRhs(PetscSNES,PetscVec*)
    int SNESGetSolution(PetscSNES,PetscVec*)
    int SNESGetSolutionUpdate(PetscSNES,PetscVec*)

    int SNESSetFunction(PetscSNES,PetscVec,PetscSNESFunction*,void*)
    int SNESGetFunction(PetscSNES,PetscVec*,PetscSNESFunction**,void**)
    int SNESSetUpdate(PetscSNES,PetscSNESUpdate*)
    int SNESDefaultUpdate(PetscSNES,PetscInt) except PETSC_ERR_PYTHON
    int SNESSetJacobian(PetscSNES,PetscMat,PetscMat,PetscSNESJacobian*,void*)
    int SNESGetJacobian(PetscSNES,PetscMat*,PetscMat*,PetscSNESJacobian**,void**)

    int SNESComputeFunction(PetscSNES,PetscVec,PetscVec)
    int SNESComputeJacobian(PetscSNES,PetscVec,PetscMat*,PetscMat*,PetscMatStructure*)

    int SNESSetTolerances(PetscSNES,PetscReal,PetscReal,PetscReal,PetscInt,PetscInt)
    int SNESGetTolerances(PetscSNES,PetscReal*,PetscReal*,PetscReal*,PetscInt*,PetscInt*)

    int SNESSetConvergenceTest(PetscSNES,PetscSNESConverged*,void*,PetscSNESCtxDel*)
    int SNESDefaultConverged(PetscSNES,PetscInt,PetscReal,PetscReal,PetscReal,
                             PetscSNESConvergedReason*,void*) except PETSC_ERR_PYTHON
    int SNESSkipConverged(PetscSNES,PetscInt,PetscReal,PetscReal,PetscReal,
                          PetscSNESConvergedReason*,void*) except PETSC_ERR_PYTHON
    int SNESSetConvergenceHistory(PetscSNES,PetscReal[],PetscInt[],PetscInt,PetscTruth)
    int SNESGetConvergenceHistory(PetscSNES,PetscReal*[],PetscInt*[],PetscInt*)
    int SNESLogConvergenceHistory(PetscSNES,PetscInt,PetscReal,PetscInt)

    int SNESMonitorSet(PetscSNES,PetscSNESMonitor*,void*,PetscSNESCtxDel*)
    int SNESMonitorCancel(PetscSNES)

    int SNESSetUp(PetscSNES)
    int SNESSolve(PetscSNES,PetscVec,PetscVec)

    int SNESSetConvergedReason(PetscSNES,PetscSNESConvergedReason)
    int SNESGetConvergedReason(PetscSNES,PetscSNESConvergedReason*)
    int SNESSetIterationNumber(PetscSNES,PetscInt)
    int SNESGetIterationNumber(PetscSNES,PetscInt*)
    int SNESSetFunctionNorm(PetscSNES,PetscScalar)
    int SNESGetFunctionNorm(PetscSNES,PetscScalar*)
    int SNESGetLinearSolveIterations(PetscSNES,PetscInt*)

    int SNESGetNumberFunctionEvals(PetscSNES,PetscInt*)
    int SNESSetMaxNonlinearStepFailures(PetscSNES,PetscInt)
    int SNESGetMaxNonlinearStepFailures(PetscSNES,PetscInt*)
    int SNESGetNonlinearStepFailures(PetscSNES,PetscInt*)
    int SNESSetMaxLinearSolveFailures(PetscSNES,PetscInt)
    int SNESGetMaxLinearSolveFailures(PetscSNES,PetscInt*)
    int SNESGetLinearSolveFailures(PetscSNES,PetscInt*)

    int SNESKSPSetUseEW(PetscSNES,PetscTruth)
    int SNESKSPGetUseEW(PetscSNES,PetscTruth*)
    int SNESKSPSetParametersEW(PetscSNES,PetscInt,PetscReal,PetscReal,
                               PetscReal,PetscReal,PetscReal,PetscReal)
    int SNESKSPGetParametersEW(PetscSNES,PetscInt*,PetscReal*,PetscReal*,
                               PetscReal*,PetscReal*,PetscReal*,PetscReal*)

cdef extern from "custom.h" nogil:
    int SNESSetUseMFFD(PetscSNES,PetscTruth)
    int SNESGetUseMFFD(PetscSNES,PetscTruth*)

    int SNESSetUseFDColoring(PetscSNES,PetscTruth)
    int SNESGetUseFDColoring(PetscSNES,PetscTruth*)

    int SNESMonitorCall(PetscSNES,PetscInt,PetscReal)
    int SNESConvergenceTestCall(PetscSNES,PetscInt,
                                PetscReal,PetscReal,PetscReal,
                                PetscSNESConvergedReason*)

# --------------------------------------------------------------------

cdef inline SNES ref_SNES(PetscSNES snes):
    cdef SNES ob = <SNES> SNES()
    PetscIncref(<PetscObject>snes)
    ob.snes = snes
    return ob

# --------------------------------------------------------------------

cdef inline object SNES_getFun(PetscSNES snes):
    return Object_getAttr(<PetscObject>snes, "__function__")

cdef int SNES_Function(PetscSNES snes,
                       PetscVec  x,
                       PetscVec  f,
                       void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Vec  Fvec = ref_Vec(f)
    (function, args, kargs) = SNES_getFun(snes)
    function(Snes, Xvec, Fvec, *args, **kargs)
    return 0

cdef inline int SNES_setFun(PetscSNES snes, PetscVec f, object fun) except -1:
    CHKERR( SNESSetFunction(snes, f, SNES_Function, NULL) )
    Object_setAttr(<PetscObject>snes, "__function__", fun)
    return 0

# --------------------------------------------------------------------

cdef inline object SNES_getUpd(PetscSNES snes):
    return Object_getAttr(<PetscObject>snes, "__update__")

cdef int SNES_Update(PetscSNES snes,
                     PetscInt its) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    (update, args, kargs) = SNES_getUpd(snes)
    update(Snes, its, *args, **kargs)
    return 0

cdef inline int SNES_setUpd(PetscSNES snes, object upd) except -1:
    if upd is None: CHKERR( SNESSetUpdate(snes, NULL) )
    else: CHKERR( SNESSetUpdate(snes, SNES_Update) )
    Object_setAttr(<PetscObject>snes, "__update__", upd)
    return 0

# --------------------------------------------------------------------

cdef inline object SNES_getJac(PetscSNES snes):
    return Object_getAttr(<PetscObject>snes, "__jacobian__")

cdef int SNES_Jacobian(PetscSNES snes,
                       PetscVec  x,
                       PetscMat  *J,
                       PetscMat  *P,
                       PetscMatStructure* s,
                       void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Mat  Jmat = ref_Mat(J[0])
    cdef Mat  Pmat = ref_Mat(P[0])
    (jacobian, args, kargs) = SNES_getJac(snes)
    retv = jacobian(Snes, Xvec, Jmat, Pmat, *args, **kargs)
    s[0] = matstructure(retv)
    cdef PetscMat Jtmp = NULL, Ptmp = NULL
    Jtmp = J[0]; J[0] = Jmat.mat; Jmat.mat = Jtmp
    Ptmp = P[0]; P[0] = Pmat.mat; Pmat.mat = Ptmp
    return 0

cdef inline int SNES_setJac(PetscSNES snes,
                            PetscMat J, PetscMat P,
                            object jac) except -1:
    CHKERR( SNESSetJacobian(snes, J, P, SNES_Jacobian, NULL) )
    Object_setAttr(<PetscObject>snes, "__jacobian__", jac)
    return 0

# --------------------------------------------------------------------

cdef inline object SNES_getCnv(PetscSNES snes):
    return Object_getAttr(<PetscObject>snes, "__converged__")

cdef int SNES_Converged(PetscSNES  snes,
                        PetscInt   iters,
                        PetscReal  xnorm,
                        PetscReal  gnorm,
                        PetscReal  fnorm,
                        PetscSNESConvergedReason *r,
                        void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    (converged, args, kargs) = SNES_getCnv(snes)
    cdef object it = iters
    cdef object xn = toReal(xnorm)
    cdef object gn = toReal(gnorm)
    cdef object fn = toReal(fnorm)
    reason = converged(Snes, it, (xn, gn, fn), *args, **kargs)
    if   reason is None:  r[0] = SNES_CONVERGED_ITERATING
    elif reason is False: r[0] = SNES_CONVERGED_ITERATING
    elif reason is True:  r[0] = SNES_CONVERGED_ITS # XXX ?
    else:                 r[0] = reason
    return 0

cdef inline int SNES_setCnv(PetscSNES snes, object cnv) except -1:
    if cnv is None: CHKERR( SNESSetConvergenceTest(
        snes, SNESDefaultConverged, NULL, NULL) )
    else: CHKERR( SNESSetConvergenceTest(snes, SNES_Converged, NULL, NULL) )
    Object_setAttr(<PetscObject>snes, "__converged__", cnv)
    return 0

# --------------------------------------------------------------------

cdef inline object SNES_getMon(PetscSNES snes):
    return Object_getAttr(<PetscObject>snes, "__monitor__")

cdef int SNES_Monitor(PetscSNES  snes,
                      PetscInt   iters,
                      PetscReal  rnorm,
                      void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef object monitorlist = SNES_getMon(snes)
    if monitorlist is None: return 0
    cdef SNES Snes = ref_SNES(snes)
    cdef object it = iters
    cdef object rn = toReal(rnorm)
    for (monitor, args, kargs) in monitorlist:
        monitor(Snes, it, rn, *args, **kargs)
    return 0

cdef inline int SNES_setMon(PetscSNES snes, object mon) except -1:
    if mon is None: return 0
    CHKERR( SNESMonitorSet(snes, SNES_Monitor, NULL, NULL) )
    cdef object monitorlist = SNES_getMon(snes)
    if monitorlist is None: monitorlist = [mon]
    else: monitorlist.append(mon)
    Object_setAttr(<PetscObject>snes, "__monitor__", monitorlist)
    return 0

cdef inline int SNES_clsMon(PetscSNES snes) except -1:
    CHKERR( SNESMonitorCancel(snes) )
    Object_setAttr(<PetscObject>snes, "__monitor__", None)
    return 0

# --------------------------------------------------------------------

cdef extern from "libpetsc4py.h":
    PetscSNESType SNESPYTHON
    int SNESPythonSetContext(PetscSNES,void*)
    int SNESPythonGetContext(PetscSNES,void**)

# --------------------------------------------------------------------
