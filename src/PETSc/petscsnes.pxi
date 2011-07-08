cdef extern from * nogil:

    ctypedef char* PetscSNESType "const char*"
    PetscSNESType SNESLS
    PetscSNESType SNESTR
    #PetscSNESType SNESPYTHON
    PetscSNESType SNESTEST
    PetscSNESType SNESPICARD
    PetscSNESType SNESKSPONLY
    PetscSNESType SNESVI

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
      SNES_DIVERGED_LINE_SEARCH
      SNES_DIVERGED_LOCAL_MIN


    ctypedef int (*PetscSNESCtxDel)(void*)

    ctypedef int (*PetscSNESInitialGuessFunction)(PetscSNES,
                                                  PetscVec,
                                                  void*) except PETSC_ERR_PYTHON
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
    int SNESDestroy(PetscSNES*)
    int SNESView(PetscSNES,PetscViewer)

    int SNESSetType(PetscSNES,PetscSNESType)
    int SNESGetType(PetscSNES,PetscSNESType*)
    int SNESSetOptionsPrefix(PetscSNES,char[])
    int SNESAppendOptionsPrefix(PetscSNES,char[])
    int SNESGetOptionsPrefix(PetscSNES,char*[])
    int SNESSetFromOptions(PetscSNES)

    int SNESGetKSP(PetscSNES,PetscKSP*)
    int SNESSetKSP(PetscSNES,PetscKSP)

    int SNESGetDM(PetscSNES,PetscDM*)
    int SNESSetDM(PetscSNES,PetscDM)

    int SNESGetRhs(PetscSNES,PetscVec*)
    int SNESGetSolution(PetscSNES,PetscVec*)
    int SNESGetSolutionUpdate(PetscSNES,PetscVec*)

    int SNESSetInitialGuess"SNESSetComputeInitialGuess"(PetscSNES,PetscSNESInitialGuessFunction,void*)
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
    int SNESSetConvergenceHistory(PetscSNES,PetscReal[],PetscInt[],PetscInt,PetscBool)
    int SNESGetConvergenceHistory(PetscSNES,PetscReal*[],PetscInt*[],PetscInt*)
    int SNESLogConvergenceHistory(PetscSNES,PetscInt,PetscReal,PetscInt)

    int SNESMonitorSet(PetscSNES,PetscSNESMonitorFunction,void*,PetscSNESCtxDel)
    int SNESMonitorCancel(PetscSNES)
    int SNESMonitor(PetscSNES,PetscInt,PetscReal)

    int SNESSetUp(PetscSNES)
    int SNESReset(PetscSNES)
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

    int SNESKSPSetUseEW(PetscSNES,PetscBool)
    int SNESKSPGetUseEW(PetscSNES,PetscBool*)
    int SNESKSPSetParametersEW(PetscSNES,PetscInt,PetscReal,PetscReal,
                               PetscReal,PetscReal,PetscReal,PetscReal)
    int SNESKSPGetParametersEW(PetscSNES,PetscInt*,PetscReal*,PetscReal*,
                               PetscReal*,PetscReal*,PetscReal*,PetscReal*)

cdef extern from "custom.h" nogil:
    int SNESSetUseMFFD(PetscSNES,PetscBool)
    int SNESGetUseMFFD(PetscSNES,PetscBool*)

    int SNESSetUseFDColoring(PetscSNES,PetscBool)
    int SNESGetUseFDColoring(PetscSNES,PetscBool*)

    int SNESConvergenceTestCall(PetscSNES,PetscInt,
                                PetscReal,PetscReal,PetscReal,
                                PetscSNESConvergedReason*)

cdef extern from "libpetsc4py.h":
    PetscSNESType SNESPYTHON
    int SNESPythonSetContext(PetscSNES,void*)
    int SNESPythonGetContext(PetscSNES,void**)
    int SNESPythonSetType(PetscSNES,char[])

# -----------------------------------------------------------------------------

cdef inline SNES ref_SNES(PetscSNES snes):
    cdef SNES ob = <SNES> SNES()
    ob.snes = snes
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

cdef int SNES_InitialGuess(
    PetscSNES snes,
    PetscVec  x,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    (initialguess, args, kargs) = Snes.get_attr('__initialguess__')
    initialguess(Snes, Xvec, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef int SNES_Function(
    PetscSNES snes,
    PetscVec  x,
    PetscVec  f,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Vec  Fvec = ref_Vec(f)
    (function, args, kargs) = Snes.get_attr('__function__')
    function(Snes, Xvec, Fvec, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef int SNES_Update(
    PetscSNES snes,
    PetscInt  its,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    (update, args, kargs) = Snes.get_attr('__update__')
    update(Snes, toInt(its), *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef int SNES_Jacobian(
    PetscSNES snes,
    PetscVec  x,
    PetscMat* J,
    PetscMat* P,
    PetscMatStructure* s,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Mat  Jmat = ref_Mat(J[0])
    cdef Mat  Pmat = ref_Mat(P[0])
    (jacobian, args, kargs) = Snes.get_attr('__jacobian__')
    retv = jacobian(Snes, Xvec, Jmat, Pmat, *args, **kargs)
    s[0] = matstructure(retv)
    cdef PetscMat Jtmp = NULL, Ptmp = NULL
    Jtmp = J[0]; J[0] = Jmat.mat; Jmat.mat = Jtmp
    Ptmp = P[0]; P[0] = Pmat.mat; Pmat.mat = Ptmp
    return 0

# -----------------------------------------------------------------------------

cdef int SNES_Converged(
    PetscSNES  snes,
    PetscInt   iters,
    PetscReal  xnorm,
    PetscReal  gnorm,
    PetscReal  fnorm,
    PetscSNESConvergedReason *r,
    void*      ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef object it = toInt(iters)
    cdef object xn = toReal(xnorm)
    cdef object gn = toReal(gnorm)
    cdef object fn = toReal(fnorm)
    (converged, args, kargs) = Snes.get_attr('__converged__')
    reason = converged(Snes, it, (xn, gn, fn), *args, **kargs)
    if   reason is None:  r[0] = SNES_CONVERGED_ITERATING
    elif reason is False: r[0] = SNES_CONVERGED_ITERATING
    elif reason is True:  r[0] = SNES_CONVERGED_ITS # XXX ?
    else:                 r[0] = reason
    return 0

# -----------------------------------------------------------------------------

cdef int SNES_Monitor(
    PetscSNES  snes,
    PetscInt   iters,
    PetscReal  rnorm,
    void*      ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef object monitorlist = Snes.get_attr('__monitor__')
    if monitorlist is None: return 0
    cdef object it = toInt(iters)
    cdef object rn = toReal(rnorm)
    for (monitor, args, kargs) in monitorlist:
        monitor(Snes, it, rn, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------
