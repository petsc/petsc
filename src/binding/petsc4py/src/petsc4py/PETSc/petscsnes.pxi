cdef extern from * nogil:

    ctypedef const char* PetscSNESType "SNESType"
    PetscSNESType SNESNEWTONLS
    PetscSNESType SNESNEWTONTR
    PetscSNESType SNESPYTHON
    PetscSNESType SNESNRICHARDSON
    PetscSNESType SNESKSPONLY
    PetscSNESType SNESKSPTRANSPOSEONLY
    PetscSNESType SNESVINEWTONRSLS
    PetscSNESType SNESVINEWTONSSLS
    PetscSNESType SNESNGMRES
    PetscSNESType SNESQN
    PetscSNESType SNESSHELL
    PetscSNESType SNESNGS
    PetscSNESType SNESNCG
    PetscSNESType SNESFAS
    PetscSNESType SNESMS
    PetscSNESType SNESNASM
    PetscSNESType SNESANDERSON
    PetscSNESType SNESASPIN
    PetscSNESType SNESCOMPOSITE
    PetscSNESType SNESPATCH

    ctypedef enum PetscSNESNormSchedule "SNESNormSchedule":
      SNES_NORM_DEFAULT
      SNES_NORM_NONE
      SNES_NORM_ALWAYS
      SNES_NORM_INITIAL_ONLY
      SNES_NORM_FINAL_ONLY
      SNES_NORM_INITIAL_FINAL_ONLY

    ctypedef enum PetscSNESConvergedReason "SNESConvergedReason":
      # iterating
      SNES_CONVERGED_ITERATING
      # converged
      SNES_CONVERGED_FNORM_ABS
      SNES_CONVERGED_FNORM_RELATIVE
      SNES_CONVERGED_SNORM_RELATIVE
      SNES_CONVERGED_ITS
      # diverged
      SNES_DIVERGED_FUNCTION_DOMAIN
      SNES_DIVERGED_FUNCTION_COUNT
      SNES_DIVERGED_LINEAR_SOLVE
      SNES_DIVERGED_FNORM_NAN
      SNES_DIVERGED_MAX_IT
      SNES_DIVERGED_LINE_SEARCH
      SNES_DIVERGED_INNER
      SNES_DIVERGED_LOCAL_MIN
      SNES_DIVERGED_DTOL
      SNES_DIVERGED_JACOBIAN_DOMAIN
      SNES_DIVERGED_TR_DELTA

    ctypedef PetscErrorCode (*PetscSNESCtxDel)(void*)

    ctypedef PetscErrorCode (*PetscSNESInitialGuessFunction)(PetscSNES,
                                                  PetscVec,
                                                  void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscSNESFunctionFunction)(PetscSNES,
                                              PetscVec,
                                              PetscVec,
                                              void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscSNESUpdateFunction)(PetscSNES,
                                            PetscInt) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscSNESJacobianFunction)(PetscSNES,
                                              PetscVec,
                                              PetscMat,
                                              PetscMat,
                                              void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscSNESObjectiveFunction)(PetscSNES,
                                               PetscVec,
                                               PetscReal*,
                                               void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscSNESConvergedFunction)(PetscSNES,
                                               PetscInt,
                                               PetscReal,
                                               PetscReal,
                                               PetscReal,
                                               PetscSNESConvergedReason*,
                                               void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscSNESMonitorFunction)(PetscSNES,
                                             PetscInt,
                                             PetscReal,
                                             void*) except PETSC_ERR_PYTHON

    PetscErrorCode SNESCreate(MPI_Comm,PetscSNES*)
    PetscErrorCode SNESDestroy(PetscSNES*)
    PetscErrorCode SNESView(PetscSNES,PetscViewer)

    PetscErrorCode SNESSetType(PetscSNES,PetscSNESType)
    PetscErrorCode SNESGetType(PetscSNES,PetscSNESType*)
    PetscErrorCode SNESSetOptionsPrefix(PetscSNES,char[])
    PetscErrorCode SNESAppendOptionsPrefix(PetscSNES,char[])
    PetscErrorCode SNESGetOptionsPrefix(PetscSNES,char*[])
    PetscErrorCode SNESSetFromOptions(PetscSNES)
    PetscErrorCode SNESSetApplicationContext(PetscSNES,void*)
    PetscErrorCode SNESGetApplicationContext(PetscSNES,void*)

    PetscErrorCode SNESGetKSP(PetscSNES,PetscKSP*)
    PetscErrorCode SNESSetKSP(PetscSNES,PetscKSP)

    PetscErrorCode SNESGetDM(PetscSNES,PetscDM*)
    PetscErrorCode SNESSetDM(PetscSNES,PetscDM)

    PetscErrorCode SNESFASSetInterpolation(PetscSNES,PetscInt,PetscMat)
    PetscErrorCode SNESFASGetInterpolation(PetscSNES,PetscInt,PetscMat*)
    PetscErrorCode SNESFASSetRestriction(PetscSNES,PetscInt,PetscMat)
    PetscErrorCode SNESFASGetRestriction(PetscSNES,PetscInt,PetscMat*)
    PetscErrorCode SNESFASSetInjection(PetscSNES,PetscInt,PetscMat)
    PetscErrorCode SNESFASGetInjection(PetscSNES,PetscInt,PetscMat*)
    PetscErrorCode SNESFASSetRScale(PetscSNES,PetscInt,PetscVec)
    PetscErrorCode SNESFASSetLevels(PetscSNES,PetscInt,MPI_Comm[])
    PetscErrorCode SNESFASGetLevels(PetscSNES,PetscInt*)
    PetscErrorCode SNESFASGetCycleSNES(PetscSNES,PetscInt,PetscSNES*)
    PetscErrorCode SNESFASGetCoarseSolve(PetscSNES,PetscSNES*)
    PetscErrorCode SNESFASGetSmoother(PetscSNES,PetscInt,PetscSNES*)
    PetscErrorCode SNESFASGetSmootherDown(PetscSNES,PetscInt,PetscSNES*)
    PetscErrorCode SNESFASGetSmootherUp(PetscSNES,PetscInt,PetscSNES*)

    PetscErrorCode SNESGetNPC(PetscSNES,PetscSNES*)
    PetscErrorCode SNESHasNPC(PetscSNES,PetscBool*)
    PetscErrorCode SNESSetNPC(PetscSNES,PetscSNES)
    PetscErrorCode SNESSetNPCSide(PetscSNES,PetscPCSide)
    PetscErrorCode SNESGetNPCSide(PetscSNES,PetscPCSide*)

    PetscErrorCode SNESGetRhs(PetscSNES,PetscVec*)
    PetscErrorCode SNESGetSolution(PetscSNES,PetscVec*)
    PetscErrorCode SNESSetSolution(PetscSNES,PetscVec)
    PetscErrorCode SNESGetSolutionUpdate(PetscSNES,PetscVec*)

    PetscErrorCode SNESSetInitialGuess"SNESSetComputeInitialGuess"(PetscSNES,PetscSNESInitialGuessFunction,void*)
    PetscErrorCode SNESSetFunction(PetscSNES,PetscVec,PetscSNESFunctionFunction,void*)
    PetscErrorCode SNESGetFunction(PetscSNES,PetscVec*,void*,void**)
    PetscErrorCode SNESSetUpdate(PetscSNES,PetscSNESUpdateFunction)
    PetscErrorCode SNESSetJacobian(PetscSNES,PetscMat,PetscMat,PetscSNESJacobianFunction,void*)
    PetscErrorCode SNESGetJacobian(PetscSNES,PetscMat*,PetscMat*,PetscSNESJacobianFunction*,void**)
    PetscErrorCode SNESSetObjective(PetscSNES,PetscSNESObjectiveFunction,void*)
    PetscErrorCode SNESGetObjective(PetscSNES,PetscSNESObjectiveFunction*,void**)

    PetscErrorCode SNESComputeFunction(PetscSNES,PetscVec,PetscVec)
    PetscErrorCode SNESComputeJacobian(PetscSNES,PetscVec,PetscMat,PetscMat)
    PetscErrorCode SNESComputeObjective(PetscSNES,PetscVec,PetscReal*)

    ctypedef PetscErrorCode (*PetscSNESNGSFunction)(PetscSNES,
                                         PetscVec,
                                         PetscVec,
                                         void*) except PETSC_ERR_PYTHON
    PetscErrorCode SNESSetNGS(PetscSNES,PetscSNESNGSFunction,void*)
    PetscErrorCode SNESGetNGS(PetscSNES,PetscSNESNGSFunction*,void**)
    PetscErrorCode SNESComputeNGS(PetscSNES,PetscVec,PetscVec)

    PetscErrorCode SNESSetNormSchedule(PetscSNES,PetscSNESNormSchedule)
    PetscErrorCode SNESGetNormSchedule(PetscSNES,PetscSNESNormSchedule*)

    PetscErrorCode SNESSetTolerances(PetscSNES,PetscReal,PetscReal,PetscReal,PetscInt,PetscInt)
    PetscErrorCode SNESGetTolerances(PetscSNES,PetscReal*,PetscReal*,PetscReal*,PetscInt*,PetscInt*)

    PetscErrorCode SNESSetConvergenceTest(PetscSNES,PetscSNESConvergedFunction,void*,PetscSNESCtxDel*)
    PetscErrorCode SNESConvergedDefault(PetscSNES,PetscInt,PetscReal,PetscReal,PetscReal,
                             PetscSNESConvergedReason*,void*) except PETSC_ERR_PYTHON
    PetscErrorCode SNESConvergedSkip(PetscSNES,PetscInt,PetscReal,PetscReal,PetscReal,
                          PetscSNESConvergedReason*,void*) except PETSC_ERR_PYTHON
    PetscErrorCode SNESSetConvergenceHistory(PetscSNES,PetscReal[],PetscInt[],PetscInt,PetscBool)
    PetscErrorCode SNESGetConvergenceHistory(PetscSNES,PetscReal*[],PetscInt*[],PetscInt*)
    PetscErrorCode SNESLogConvergenceHistory(PetscSNES,PetscReal,PetscInt)

    PetscErrorCode SNESMonitorSet(PetscSNES,PetscSNESMonitorFunction,void*,PetscSNESCtxDel)
    PetscErrorCode SNESMonitorCancel(PetscSNES)
    PetscErrorCode SNESMonitor(PetscSNES,PetscInt,PetscReal)

    PetscErrorCode SNESSetUp(PetscSNES)
    PetscErrorCode SNESReset(PetscSNES)
    PetscErrorCode SNESSolve(PetscSNES,PetscVec,PetscVec)

    PetscErrorCode SNESSetConvergedReason(PetscSNES,PetscSNESConvergedReason)
    PetscErrorCode SNESGetConvergedReason(PetscSNES,PetscSNESConvergedReason*)
    PetscErrorCode SNESSetErrorIfNotConverged(PetscSNES,PetscBool);
    PetscErrorCode SNESGetErrorIfNotConverged(PetscSNES,PetscBool*);
    PetscErrorCode SNESSetIterationNumber(PetscSNES,PetscInt)
    PetscErrorCode SNESGetIterationNumber(PetscSNES,PetscInt*)
    PetscErrorCode SNESSetForceIteration(PetscSNES,PetscBool)
    PetscErrorCode SNESSetFunctionNorm(PetscSNES,PetscReal)
    PetscErrorCode SNESGetFunctionNorm(PetscSNES,PetscReal*)
    PetscErrorCode SNESGetLinearSolveIterations(PetscSNES,PetscInt*)
    PetscErrorCode SNESSetCountersReset(PetscSNES,PetscBool)

    PetscErrorCode SNESGetNumberFunctionEvals(PetscSNES,PetscInt*)
    PetscErrorCode SNESSetMaxNonlinearStepFailures(PetscSNES,PetscInt)
    PetscErrorCode SNESGetMaxNonlinearStepFailures(PetscSNES,PetscInt*)
    PetscErrorCode SNESGetNonlinearStepFailures(PetscSNES,PetscInt*)
    PetscErrorCode SNESSetMaxLinearSolveFailures(PetscSNES,PetscInt)
    PetscErrorCode SNESGetMaxLinearSolveFailures(PetscSNES,PetscInt*)
    PetscErrorCode SNESGetLinearSolveFailures(PetscSNES,PetscInt*)

    PetscErrorCode SNESKSPSetUseEW(PetscSNES,PetscBool)
    PetscErrorCode SNESKSPGetUseEW(PetscSNES,PetscBool*)
    PetscErrorCode SNESKSPSetParametersEW(PetscSNES,PetscInt,PetscReal,PetscReal,
                               PetscReal,PetscReal,PetscReal,PetscReal)
    PetscErrorCode SNESKSPGetParametersEW(PetscSNES,PetscInt*,PetscReal*,PetscReal*,
                               PetscReal*,PetscReal*,PetscReal*,PetscReal*)

    PetscErrorCode SNESVISetVariableBounds(PetscSNES,PetscVec,PetscVec)
    #ctypedef PetscErrorCode (*PetscSNESVariableBoundsFunction)(PetscSNES,PetscVec,PetscVec)
    #int SNESVISetComputeVariableBounds(PetscSNES,PetscSNESVariableBoundsFunction)
    PetscErrorCode SNESVIGetInactiveSet(PetscSNES, PetscIS*)

    PetscErrorCode SNESCompositeGetSNES(PetscSNES,PetscInt,PetscSNES*)
    PetscErrorCode SNESCompositeGetNumber(PetscSNES,PetscInt*)
    PetscErrorCode SNESNASMGetSNES(PetscSNES,PetscInt,PetscSNES*)
    PetscErrorCode SNESNASMGetNumber(PetscSNES,PetscInt*)

    PetscErrorCode SNESPatchSetCellNumbering(PetscSNES, PetscSection)
    PetscErrorCode SNESPatchSetDiscretisationInfo(PetscSNES, PetscInt, PetscDM*, PetscInt*, PetscInt*, const PetscInt**, const PetscInt*, PetscInt, const PetscInt*, PetscInt, const PetscInt*)
    PetscErrorCode SNESPatchSetComputeOperator(PetscSNES, PetscPCPatchComputeOperator, void*)
    PetscErrorCode SNESPatchSetComputeFunction(PetscSNES, PetscPCPatchComputeFunction, void*)
    PetscErrorCode SNESPatchSetConstructType(PetscSNES, PetscPCPatchConstructType, PetscPCPatchConstructOperator, void*)

    PetscErrorCode SNESPythonSetType(PetscSNES,char[])
    PetscErrorCode SNESPythonGetType(PetscSNES,char*[])

cdef extern from * nogil: # custom.h
    PetscErrorCode SNESSetUseMFFD(PetscSNES,PetscBool)
    PetscErrorCode SNESGetUseMFFD(PetscSNES,PetscBool*)

    PetscErrorCode SNESSetUseFDColoring(PetscSNES,PetscBool)
    PetscErrorCode SNESGetUseFDColoring(PetscSNES,PetscBool*)

    PetscErrorCode SNESConvergenceTestCall(PetscSNES,PetscInt,
                                PetscReal,PetscReal,PetscReal,
                                PetscSNESConvergedReason*)

    ctypedef const char* PetscSNESLineSearchType "SNESLineSearchType"
    PetscSNESLineSearchType SNESLINESEARCHBT
    PetscSNESLineSearchType SNESLINESEARCHNLEQERR
    PetscSNESLineSearchType SNESLINESEARCHBASIC
    PetscSNESLineSearchType SNESLINESEARCHNONE
    PetscSNESLineSearchType SNESLINESEARCHL2
    PetscSNESLineSearchType SNESLINESEARCHCP
    PetscSNESLineSearchType SNESLINESEARCHSHELL
    PetscSNESLineSearchType SNESLINESEARCHNCGLINEAR

    PetscErrorCode SNESGetLineSearch(PetscSNES,PetscSNESLineSearch*)
    PetscErrorCode SNESLineSearchSetFromOptions(PetscSNESLineSearch)
    PetscErrorCode SNESLineSearchApply(PetscSNESLineSearch,PetscVec,PetscVec,PetscReal*,PetscVec)
    PetscErrorCode SNESLineSearchGetNorms(PetscSNESLineSearch,PetscReal*,PetscReal*,PetscReal*)
    PetscErrorCode SNESLineSearchDestroy(PetscSNESLineSearch*)

    ctypedef PetscErrorCode (*PetscSNESPreCheckFunction)(PetscSNESLineSearch,
                                              PetscVec,PetscVec,
                                              PetscBool*,
                                              void*) except PETSC_ERR_PYTHON
    PetscErrorCode SNESLineSearchSetPreCheck(PetscSNESLineSearch,PetscSNESPreCheckFunction,void*)
    PetscErrorCode SNESLineSearchGetSNES(PetscSNESLineSearch,PetscSNES*)

# -----------------------------------------------------------------------------

cdef inline SNES ref_SNES(PetscSNES snes):
    cdef SNES ob = <SNES> SNES()
    ob.snes = snes
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

cdef PetscErrorCode SNES_InitialGuess(
    PetscSNES snes,
    PetscVec  x,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef object context = Snes.get_attr('__initialguess__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (initialguess, args, kargs) = context
    initialguess(Snes, Xvec, *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode SNES_PreCheck(
    PetscSNESLineSearch linesearch,
    PetscVec  x,
    PetscVec  y,
    PetscBool *changed,
    void* ctx
    ) except PETSC_ERR_PYTHON with gil:
    cdef PetscSNES snes = NULL;
    CHKERR( SNESLineSearchGetSNES(linesearch, &snes) );
    cdef object b = False
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Vec  Yvec = ref_Vec(y)
    cdef object context = Snes.get_attr('__precheck__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (precheck, args, kargs) = context
    b = precheck(Xvec, Yvec, *args, **kargs)
    changed[0] = asBool(b)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------


cdef PetscErrorCode SNES_Function(
    PetscSNES snes,
    PetscVec  x,
    PetscVec  f,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Vec  Fvec = ref_Vec(f)
    cdef object context = Snes.get_attr('__function__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (function, args, kargs) = context
    function(Snes, Xvec, Fvec, *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode SNES_Update(
    PetscSNES snes,
    PetscInt  its,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef object context = Snes.get_attr('__update__')
    assert context is not None and type(context) is tuple # sanity check
    (update, args, kargs) = context
    update(Snes, toInt(its), *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode SNES_Jacobian(
    PetscSNES snes,
    PetscVec  x,
    PetscMat  J,
    PetscMat  P,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Mat  Jmat = ref_Mat(J)
    cdef Mat  Pmat = ref_Mat(P)
    cdef object context = Snes.get_attr('__jacobian__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(Snes, Xvec, Jmat, Pmat, *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode SNES_Objective(
    PetscSNES  snes,
    PetscVec   x,
    PetscReal *o,
    void*      ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef object context = Snes.get_attr('__objective__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (objective, args, kargs) = context
    obj = objective(Snes, Xvec, *args, **kargs)
    o[0] = asReal(obj)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode SNES_NGS(
    PetscSNES snes,
    PetscVec  x,
    PetscVec  b,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Vec  Bvec = ref_Vec(b)
    cdef object context = Snes.get_attr('__ngs__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (ngs, args, kargs) = context
    ngs(Snes, Xvec, Bvec, *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode SNES_Converged(
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
    cdef object context = Snes.get_attr('__converged__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (converged, args, kargs) = context
    reason = converged(Snes, it, (xn, gn, fn), *args, **kargs)
    if   reason is None:  r[0] = SNES_CONVERGED_ITERATING
    elif reason is False: r[0] = SNES_CONVERGED_ITERATING
    elif reason is True:  r[0] = SNES_CONVERGED_ITS # XXX ?
    else:                 r[0] = reason
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode SNES_Monitor(
    PetscSNES  snes,
    PetscInt   iters,
    PetscReal  rnorm,
    void*      ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef object monitorlist = Snes.get_attr('__monitor__')
    if monitorlist is None: return PETSC_SUCCESS
    cdef object it = toInt(iters)
    cdef object rn = toReal(rnorm)
    for (monitor, args, kargs) in monitorlist:
        monitor(Snes, it, rn, *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------
