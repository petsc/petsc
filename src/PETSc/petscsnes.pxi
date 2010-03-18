cdef extern from "petscsnes.h" nogil:

    ctypedef char* PetscSNESType "const char*"
    PetscSNESType SNESLS
    PetscSNESType SNESTR
    PetscSNESType SNESPICARD
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


    ctypedef int (*PetscSNESCtxDel)(void*)

    ctypedef int (*PetscSNESFunctionFunction)(PetscSNES,
                                              PetscVec,
                                              PetscVec,
                                              void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscSNESUpdateFunction)(PetscSNES,
                                            PetscInt) except PETSC_ERR_PYTHON

    ctypedef int (*PetscSNESJacobianFunction)(PetscSNES,
                                              PetscVec,
                                              PetscMat*,
                                              PetscMat*,
                                              PetscMatStructure*,
                                              void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscSNESConvergedFunction)(PetscSNES,
                                               PetscInt,
                                               PetscReal,
                                               PetscReal,
                                               PetscReal,
                                               PetscSNESConvergedReason*,
                                               void*)  except PETSC_ERR_PYTHON

    ctypedef int (*PetscSNESMonitorFunction)(PetscSNES,
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

    int SNESSetFunction(PetscSNES,PetscVec,PetscSNESFunctionFunction,void*)
    int SNESGetFunction(PetscSNES,PetscVec*,PetscSNESFunctionFunction*,void**)
    int SNESSetUpdate(PetscSNES,PetscSNESUpdateFunction)
    int SNESDefaultUpdate(PetscSNES,PetscInt) except PETSC_ERR_PYTHON
    int SNESSetJacobian(PetscSNES,PetscMat,PetscMat,PetscSNESJacobianFunction,void*)
    int SNESGetJacobian(PetscSNES,PetscMat*,PetscMat*,PetscSNESJacobianFunction*,void**)

    int SNESComputeFunction(PetscSNES,PetscVec,PetscVec)
    int SNESComputeJacobian(PetscSNES,PetscVec,PetscMat*,PetscMat*,PetscMatStructure*)

    int SNESSetTolerances(PetscSNES,PetscReal,PetscReal,PetscReal,PetscInt,PetscInt)
    int SNESGetTolerances(PetscSNES,PetscReal*,PetscReal*,PetscReal*,PetscInt*,PetscInt*)

    int SNESSetConvergenceTest(PetscSNES,PetscSNESConvergedFunction,void*,PetscSNESCtxDel*)
    int SNESDefaultConverged(PetscSNES,PetscInt,PetscReal,PetscReal,PetscReal,
                             PetscSNESConvergedReason*,void*) except PETSC_ERR_PYTHON
    int SNESSkipConverged(PetscSNES,PetscInt,PetscReal,PetscReal,PetscReal,
                          PetscSNESConvergedReason*,void*) except PETSC_ERR_PYTHON
    int SNESSetConvergenceHistory(PetscSNES,PetscReal[],PetscInt[],PetscInt,PetscTruth)
    int SNESGetConvergenceHistory(PetscSNES,PetscReal*[],PetscInt*[],PetscInt*)
    int SNESLogConvergenceHistory(PetscSNES,PetscInt,PetscReal,PetscInt)

    int SNESMonitorSet(PetscSNES,PetscSNESMonitorFunction,void*,PetscSNESCtxDel)
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

# -----------------------------------------------------------------------------

cdef inline object SNES_getFunction(PetscSNES snes):
    return Object_getAttr(<PetscObject>snes, '__function__')

cdef inline int SNES_setFunction(PetscSNES snes,
                                 PetscVec f,
                                 object function) except -1:
    CHKERR( SNESSetFunction(snes, f, SNES_Function, NULL) )
    Object_setAttr(<PetscObject>snes, '__function__', function)
    return 0

cdef int SNES_Function(PetscSNES snes,
                       PetscVec  x,
                       PetscVec  f,
                       void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Vec  Fvec = ref_Vec(f)
    (function, args, kargs) = SNES_getFunction(snes)
    function(Snes, Xvec, Fvec, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef inline object SNES_getUpdate(PetscSNES snes):
    return Object_getAttr(<PetscObject>snes, '__update__')

cdef inline int SNES_setUpdate(PetscSNES snes, object update) except -1:
    if update is not None:
        CHKERR( SNESSetUpdate(snes, SNES_Update) )
    else:
        CHKERR( SNESSetUpdate(snes, NULL) )
    Object_setAttr(<PetscObject>snes, '__update__', update)
    return 0

cdef int SNES_Update(PetscSNES snes,
                     PetscInt its) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    (update, args, kargs) = SNES_getUpdate(snes)
    update(Snes, its, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef inline object SNES_getJacobian(PetscSNES snes):
    return Object_getAttr(<PetscObject>snes, '__jacobian__')

cdef inline int SNES_setJacobian(PetscSNES snes,
                                 PetscMat J, PetscMat P,
                                 object jacobian) except -1:
    CHKERR( SNESSetJacobian(snes, J, P, SNES_Jacobian, NULL) )
    Object_setAttr(<PetscObject>snes, '__jacobian__', jacobian)
    return 0

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
    (jacobian, args, kargs) = SNES_getJacobian(snes)
    retv = jacobian(Snes, Xvec, Jmat, Pmat, *args, **kargs)
    s[0] = matstructure(retv)
    cdef PetscMat Jtmp = NULL, Ptmp = NULL
    Jtmp = J[0]; J[0] = Jmat.mat; Jmat.mat = Jtmp
    Ptmp = P[0]; P[0] = Pmat.mat; Pmat.mat = Ptmp
    return 0

# -----------------------------------------------------------------------------

cdef inline object SNES_getConverged(PetscSNES snes):
    return Object_getAttr(<PetscObject>snes, '__converged__')

cdef inline int SNES_setConverged(PetscSNES snes, object converged) except -1:
    if converged is not None:
        CHKERR( SNESSetConvergenceTest(
                snes, SNES_Converged, NULL, NULL) )
    else:
        CHKERR( SNESSetConvergenceTest(
                snes, SNESDefaultConverged, NULL, NULL) )
    Object_setAttr(<PetscObject>snes, '__converged__', converged)
    return 0

cdef int SNES_Converged(PetscSNES  snes,
                        PetscInt   iters,
                        PetscReal  xnorm,
                        PetscReal  gnorm,
                        PetscReal  fnorm,
                        PetscSNESConvergedReason *r,
                        void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    (converged, args, kargs) = SNES_getConverged(snes)
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

# --------------------------------------------------------------------

cdef inline object SNES_getMonitor(PetscSNES snes):
    return Object_getAttr(<PetscObject>snes, '__monitor__')

cdef inline int SNES_setMonitor(PetscSNES snes, object monitor) except -1:
    CHKERR( SNESMonitorSet(snes, SNES_Monitor, NULL, NULL) )
    cdef object monitorlist = SNES_getMonitor(snes)
    if monitor is None: monitorlist = None
    elif monitorlist is None: monitorlist = [monitor]
    else: monitorlist.append(monitor)
    Object_setAttr(<PetscObject>snes, '__monitor__', monitorlist)
    return 0

cdef inline int SNES_delMonitor(PetscSNES snes) except -1:
    Object_setAttr(<PetscObject>snes, '__monitor__', None)
    return 0

cdef int SNES_Monitor(PetscSNES  snes,
                      PetscInt   iters,
                      PetscReal  rnorm,
                      void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef object monitorlist = SNES_getMonitor(snes)
    if monitorlist is None: return 0
    cdef SNES Snes = ref_SNES(snes)
    cdef object it = iters
    cdef object rn = toReal(rnorm)
    for (monitor, args, kargs) in monitorlist:
        monitor(Snes, it, rn, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef extern from "libpetsc4py.h":
    PetscSNESType SNESPYTHON
    int SNESPythonSetContext(PetscSNES,void*)
    int SNESPythonGetContext(PetscSNES,void**)
    int SNESPythonSetType(PetscSNES,char[])

# -----------------------------------------------------------------------------
