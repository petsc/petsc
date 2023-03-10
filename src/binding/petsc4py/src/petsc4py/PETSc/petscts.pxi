cdef extern from * nogil:

    ctypedef const char* PetscTSType "TSType"
    PetscTSType TSEULER
    PetscTSType TSBEULER
    PetscTSType TSBASICSYMPLECTIC
    PetscTSType TSPSEUDO
    PetscTSType TSCN
    PetscTSType TSSUNDIALS
    PetscTSType TSRK
    PetscTSType TSPYTHON
    PetscTSType TSTHETA
    PetscTSType TSALPHA
    PetscTSType TSALPHA2
    PetscTSType TSGLLE
    PetscTSType TSGLEE
    PetscTSType TSSSP
    PetscTSType TSARKIMEX
    PetscTSType TSROSW
    PetscTSType TSEIMEX
    PetscTSType TSMIMEX
    PetscTSType TSBDF
    PetscTSType TSRADAU5
    PetscTSType TSMPRK
    PetscTSType TSDISCGRAD

    ctypedef enum PetscTSProblemType "TSProblemType":
      TS_LINEAR
      TS_NONLINEAR

    ctypedef enum PetscTSEquationType "TSEquationType":
      TS_EQ_UNSPECIFIED
      TS_EQ_EXPLICIT
      TS_EQ_ODE_EXPLICIT
      TS_EQ_DAE_SEMI_EXPLICIT_INDEX1
      TS_EQ_DAE_SEMI_EXPLICIT_INDEX2
      TS_EQ_DAE_SEMI_EXPLICIT_INDEX3
      TS_EQ_DAE_SEMI_EXPLICIT_INDEXHI
      TS_EQ_IMPLICIT
      TS_EQ_ODE_IMPLICIT
      TS_EQ_DAE_IMPLICIT_INDEX1
      TS_EQ_DAE_IMPLICIT_INDEX2
      TS_EQ_DAE_IMPLICIT_INDEX3
      TS_EQ_DAE_IMPLICIT_INDEXHI

    ctypedef enum PetscTSConvergedReason "TSConvergedReason":
      # iterating
      TS_CONVERGED_ITERATING
      # converged
      TS_CONVERGED_TIME
      TS_CONVERGED_ITS
      TS_CONVERGED_USER
      TS_CONVERGED_EVENT
      # diverged
      TS_DIVERGED_NONLINEAR_SOLVE
      TS_DIVERGED_STEP_REJECTED

    ctypedef enum PetscTSExactFinalTimeOption "TSExactFinalTimeOption":
      TS_EXACTFINALTIME_UNSPECIFIED
      TS_EXACTFINALTIME_STEPOVER
      TS_EXACTFINALTIME_INTERPOLATE
      TS_EXACTFINALTIME_MATCHSTEP

    ctypedef PetscErrorCode PetscTSCtxDel(void*)

    ctypedef PetscErrorCode (*PetscTSFunctionFunction)(PetscTS,
                                            PetscReal,
                                            PetscVec,
                                            PetscVec,
                                            void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscTSJacobianFunction)(PetscTS,
                                            PetscReal,
                                            PetscVec,
                                            PetscMat,
                                            PetscMat,
                                            void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscTSIFunctionFunction)(PetscTS,
                                             PetscReal,
                                             PetscVec,
                                             PetscVec,
                                             PetscVec,
                                             void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscTSIJacobianFunction)(PetscTS,
                                             PetscReal,
                                             PetscVec,
                                             PetscVec,
                                             PetscReal,
                                             PetscMat,
                                             PetscMat,
                                             void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscTSIJacobianPFunction)(PetscTS,
                                             PetscReal,
                                             PetscVec,
                                             PetscVec,
                                             PetscReal,
                                             PetscMat,
                                             void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscTSI2FunctionFunction)(PetscTS,
                                              PetscReal,
                                              PetscVec,
                                              PetscVec,
                                              PetscVec,
                                              PetscVec,
                                              void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscTSI2JacobianFunction)(PetscTS,
                                              PetscReal,
                                              PetscVec,
                                              PetscVec,
                                              PetscVec,
                                              PetscReal,
                                              PetscReal,
                                              PetscMat,
                                              PetscMat,
                                              void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscTSMonitorFunction)(PetscTS,
                                           PetscInt,
                                           PetscReal,
                                           PetscVec,
                                           void*) except PETSC_ERR_PYTHON

    ctypedef PetscErrorCode (*PetscTSPreStepFunction)  (PetscTS) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscTSPostStepFunction) (PetscTS) except PETSC_ERR_PYTHON

    PetscErrorCode TSCreate(MPI_Comm comm,PetscTS*)
    PetscErrorCode TSClone(PetscTS,PetscTS*)
    PetscErrorCode TSDestroy(PetscTS*)
    PetscErrorCode TSView(PetscTS,PetscViewer)
    PetscErrorCode TSLoad(PetscTS,PetscViewer)

    PetscErrorCode TSSetProblemType(PetscTS,PetscTSProblemType)
    PetscErrorCode TSGetProblemType(PetscTS,PetscTSProblemType*)
    PetscErrorCode TSSetEquationType(PetscTS,PetscTSEquationType)
    PetscErrorCode TSGetEquationType(PetscTS,PetscTSEquationType*)
    PetscErrorCode TSSetType(PetscTS,PetscTSType)
    PetscErrorCode TSGetType(PetscTS,PetscTSType*)

    PetscErrorCode TSSetOptionsPrefix(PetscTS,char[])
    PetscErrorCode TSAppendOptionsPrefix(PetscTS,char[])
    PetscErrorCode TSGetOptionsPrefix(PetscTS,char*[])
    PetscErrorCode TSSetFromOptions(PetscTS)

    PetscErrorCode TSSetSolution(PetscTS,PetscVec)
    PetscErrorCode TSGetSolution(PetscTS,PetscVec*)
    PetscErrorCode TS2SetSolution(PetscTS,PetscVec,PetscVec)
    PetscErrorCode TS2GetSolution(PetscTS,PetscVec*,PetscVec*)

    PetscErrorCode TSGetRHSFunction(PetscTS,PetscVec*,PetscTSFunctionFunction*,void*)
    PetscErrorCode TSGetRHSJacobian(PetscTS,PetscMat*,PetscMat*,PetscTSJacobianFunction*,void**)
    PetscErrorCode TSSetRHSFunction(PetscTS,PetscVec,PetscTSFunctionFunction,void*)
    PetscErrorCode TSSetRHSJacobian(PetscTS,PetscMat,PetscMat,PetscTSJacobianFunction,void*)
    PetscErrorCode TSSetIFunction(PetscTS,PetscVec,PetscTSIFunctionFunction,void*)
    PetscErrorCode TSSetIJacobian(PetscTS,PetscMat,PetscMat,PetscTSIJacobianFunction,void*)
    PetscErrorCode TSSetIJacobianP(PetscTS,PetscMat,PetscTSIJacobianPFunction,void*)
    PetscErrorCode TSGetIFunction(PetscTS,PetscVec*,PetscTSIFunctionFunction*,void*)
    PetscErrorCode TSGetIJacobian(PetscTS,PetscMat*,PetscMat*,PetscTSIJacobianFunction*,void**)
    PetscErrorCode TSSetI2Function(PetscTS,PetscVec,PetscTSI2FunctionFunction,void*)
    PetscErrorCode TSSetI2Jacobian(PetscTS,PetscMat,PetscMat,PetscTSI2JacobianFunction,void*)
    PetscErrorCode TSGetI2Function(PetscTS,PetscVec*,PetscTSI2FunctionFunction*,void**)
    PetscErrorCode TSGetI2Jacobian(PetscTS,PetscMat*,PetscMat*,PetscTSI2JacobianFunction*,void**)

    PetscErrorCode TSGetKSP(PetscTS,PetscKSP*)
    PetscErrorCode TSGetSNES(PetscTS,PetscSNES*)

    PetscErrorCode TSGetDM(PetscTS,PetscDM*)
    PetscErrorCode TSSetDM(PetscTS,PetscDM)

    PetscErrorCode TSComputeRHSFunction(PetscTS,PetscReal,PetscVec,PetscVec)
    PetscErrorCode TSComputeRHSFunctionLinear(PetscTS,PetscReal,PetscVec,PetscVec,void*)
    PetscErrorCode TSComputeRHSJacobian(PetscTS,PetscReal,PetscVec,PetscMat,PetscMat)
    PetscErrorCode TSComputeRHSJacobianConstant(PetscTS,PetscReal,PetscVec,PetscMat,PetscMat,void*)
    PetscErrorCode TSComputeIFunction(PetscTS,PetscReal,PetscVec,PetscVec,PetscVec,PetscBool)
    PetscErrorCode TSComputeIJacobian(PetscTS,PetscReal,PetscVec,PetscVec,PetscReal,PetscMat,PetscMat,PetscBool)
    PetscErrorCode TSComputeIJacobianP(PetscTS,PetscReal,PetscVec,PetscVec,PetscReal,PetscMat,PetscBool)
    PetscErrorCode TSComputeI2Function(PetscTS,PetscReal,PetscVec,PetscVec,PetscVec,PetscVec)
    PetscErrorCode TSComputeI2Jacobian(PetscTS,PetscReal,PetscVec,PetscVec,PetscVec,PetscReal,PetscReal,PetscMat,PetscMat)

    PetscErrorCode TSSetTime(PetscTS,PetscReal)
    PetscErrorCode TSGetTime(PetscTS,PetscReal*)
    PetscErrorCode TSGetPrevTime(PetscTS,PetscReal*)
    PetscErrorCode TSGetSolveTime(PetscTS,PetscReal*)
    PetscErrorCode TSSetTimeStep(PetscTS,PetscReal)
    PetscErrorCode TSGetTimeStep(PetscTS,PetscReal*)
    PetscErrorCode TSSetStepNumber(PetscTS,PetscInt)
    PetscErrorCode TSGetStepNumber(PetscTS,PetscInt*)
    PetscErrorCode TSSetMaxSteps(PetscTS,PetscInt)
    PetscErrorCode TSGetMaxSteps(PetscTS,PetscInt*)
    PetscErrorCode TSSetMaxTime(PetscTS,PetscReal)
    PetscErrorCode TSGetMaxTime(PetscTS,PetscReal*)
    PetscErrorCode TSSetExactFinalTime(PetscTS,PetscTSExactFinalTimeOption)
    PetscErrorCode TSSetTimeSpan(PetscTS,PetscInt,PetscReal*)
    PetscErrorCode TSGetTimeSpan(PetscTS,PetscInt*,const PetscReal**)
    PetscErrorCode TSGetTimeSpanSolutions(PetscTS,PetscInt*,PetscVec**)

    PetscErrorCode TSSetConvergedReason(PetscTS,PetscTSConvergedReason)
    PetscErrorCode TSGetConvergedReason(PetscTS,PetscTSConvergedReason*)
    PetscErrorCode TSGetSNESIterations(PetscTS,PetscInt*)
    PetscErrorCode TSGetKSPIterations(PetscTS,PetscInt*)
    PetscErrorCode TSGetStepRejections(PetscTS,PetscInt*)
    PetscErrorCode TSSetMaxStepRejections(PetscTS,PetscInt)
    PetscErrorCode TSGetSNESFailures(PetscTS,PetscInt*)
    PetscErrorCode TSSetMaxSNESFailures(PetscTS,PetscInt)
    PetscErrorCode TSSetErrorIfStepFails(PetscTS,PetscBool)
    PetscErrorCode TSSetTolerances(PetscTS,PetscReal,PetscVec,PetscReal,PetscVec)
    PetscErrorCode TSGetTolerances(PetscTS,PetscReal*,PetscVec*,PetscReal*,PetscVec*)

    PetscErrorCode TSMonitorSet(PetscTS,PetscTSMonitorFunction,void*,PetscTSCtxDel*)
    PetscErrorCode TSMonitorCancel(PetscTS)
    PetscErrorCode TSMonitor(PetscTS,PetscInt,PetscReal,PetscVec)

    ctypedef PetscErrorCode (*PetscTSEventHandler)(PetscTS,PetscReal,PetscVec,PetscScalar[],void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscTSPostEvent)(PetscTS,PetscInt,PetscInt[],PetscReal,PetscVec, PetscBool, void*) except PETSC_ERR_PYTHON

    PetscErrorCode TSSetEventHandler(PetscTS, PetscInt, PetscInt[], PetscBool[], PetscTSEventHandler, PetscTSPostEvent, void*)
    PetscErrorCode TSSetEventTolerances(PetscTS, PetscReal, PetscReal[])
    PetscErrorCode TSGetNumEvents(PetscTS, PetscInt*)

    ctypedef PetscErrorCode (*PetscTSAdjointR)(PetscTS,PetscReal,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscTSAdjointDRDY)(PetscTS,PetscReal,PetscVec,PetscVec[],void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscTSAdjointDRDP)(PetscTS,PetscReal,PetscVec,PetscVec[],void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscTSRHSJacobianP)(PetscTS,PetscReal,PetscVec,PetscMat,void*) except PETSC_ERR_PYTHON

    PetscErrorCode TSSetSaveTrajectory(PetscTS)
    PetscErrorCode TSRemoveTrajectory(PetscTS)
    PetscErrorCode TSSetCostGradients(PetscTS,PetscInt,PetscVec*,PetscVec*)
    PetscErrorCode TSGetCostGradients(PetscTS,PetscInt*,PetscVec**,PetscVec**)
    PetscErrorCode TSCreateQuadratureTS(PetscTS,PetscBool,PetscTS*)
    PetscErrorCode TSGetQuadratureTS(PetscTS,PetscBool*,PetscTS*)
    PetscErrorCode TSGetCostIntegral(PetscTS,PetscVec*)

    PetscErrorCode TSSetRHSJacobianP(PetscTS,PetscMat,PetscTSRHSJacobianP,void*)
    PetscErrorCode TSComputeRHSJacobianP(PetscTS,PetscReal,PetscVec,PetscMat)

    PetscErrorCode TSAdjointSolve(PetscTS)
    PetscErrorCode TSAdjointSetSteps(PetscTS,PetscInt)
    PetscErrorCode TSAdjointStep(PetscTS)
    PetscErrorCode TSAdjointSetUp(PetscTS)
    PetscErrorCode TSAdjointReset(PetscTS)
    PetscErrorCode TSAdjointComputeDRDPFunction(PetscTS,PetscReal,PetscVec,PetscVec*)
    PetscErrorCode TSAdjointComputeDRDYFunction(PetscTS,PetscReal,PetscVec,PetscVec*)
    PetscErrorCode TSAdjointCostIntegral(PetscTS)

    PetscErrorCode TSForwardSetSensitivities(PetscTS,PetscInt,PetscVec*,PetscInt,PetscVec*)
    PetscErrorCode TSForwardGetSensitivities(PetscTS,PetscInt*,PetscVec**,PetscInt*,PetscVec**)
    PetscErrorCode TSForwardSetIntegralGradients(PetscTS,PetscInt,PetscVec *,PetscVec *)
    PetscErrorCode TSForwardGetIntegralGradients(PetscTS,PetscInt*,PetscVec **,PetscVec **)
    PetscErrorCode TSForwardSetRHSJacobianP(PetscTS,PetscVec*,PetscTSCostIntegrandFunction,void*)
    PetscErrorCode TSForwardComputeRHSJacobianP(PetscTS,PetscReal,PetscVec,PetscVec*)
    PetscErrorCode TSForwardSetUp(PetscTS)
    PetscErrorCode TSForwardCostIntegral(PetscTS)
    PetscErrorCode TSForwardStep(PetscTS)

    PetscErrorCode TSSetPreStep(PetscTS, PetscTSPreStepFunction)
    PetscErrorCode TSSetPostStep(PetscTS, PetscTSPostStepFunction)

    PetscErrorCode TSSetUp(PetscTS)
    PetscErrorCode TSReset(PetscTS)
    PetscErrorCode TSStep(PetscTS)
    PetscErrorCode TSRestartStep(PetscTS)
    PetscErrorCode TSRollBack(PetscTS)
    PetscErrorCode TSSolve(PetscTS,PetscVec)
    PetscErrorCode TSInterpolate(PetscTS,PetscReal,PetscVec)
    PetscErrorCode TSPreStage(PetscTS,PetscReal)
    PetscErrorCode TSPostStage(PetscTS,PetscReal,PetscInt,PetscVec*)

    PetscErrorCode TSThetaSetTheta(PetscTS,PetscReal)
    PetscErrorCode TSThetaGetTheta(PetscTS,PetscReal*)
    PetscErrorCode TSThetaSetEndpoint(PetscTS,PetscBool)
    PetscErrorCode TSThetaGetEndpoint(PetscTS,PetscBool*)

    PetscErrorCode TSAlphaSetRadius(PetscTS,PetscReal)
    PetscErrorCode TSAlphaSetParams(PetscTS,PetscReal,PetscReal,PetscReal)
    PetscErrorCode TSAlphaGetParams(PetscTS,PetscReal*,PetscReal*,PetscReal*)

    ctypedef const char* PetscTSRKType "TSRKType"
    PetscTSRKType TSRK1FE
    PetscTSRKType TSRK2A
    PetscTSRKType TSRK2B
    PetscTSRKType TSRK3
    PetscTSRKType TSRK3BS
    PetscTSRKType TSRK4
    PetscTSRKType TSRK5F
    PetscTSRKType TSRK5DP
    PetscTSRKType TSRK5BS
    PetscTSRKType TSRK6VR
    PetscTSRKType TSRK7VR
    PetscTSRKType TSRK8VR

    PetscErrorCode TSRKGetType(PetscTS ts,PetscTSRKType*)
    PetscErrorCode TSRKSetType(PetscTS ts,PetscTSRKType)

    ctypedef const char* PetscTSARKIMEXType "TSARKIMEXType"
    PetscTSARKIMEXType TSARKIMEX1BEE
    PetscTSARKIMEXType TSARKIMEXA2
    PetscTSARKIMEXType TSARKIMEXL2
    PetscTSARKIMEXType TSARKIMEXARS122
    PetscTSARKIMEXType TSARKIMEX2C
    PetscTSARKIMEXType TSARKIMEX2D
    PetscTSARKIMEXType TSARKIMEX2E
    PetscTSARKIMEXType TSARKIMEXPRSSP2
    PetscTSARKIMEXType TSARKIMEX3
    PetscTSARKIMEXType TSARKIMEXBPR3
    PetscTSARKIMEXType TSARKIMEXARS443
    PetscTSARKIMEXType TSARKIMEX4
    PetscTSARKIMEXType TSARKIMEX5

    PetscErrorCode TSARKIMEXGetType(PetscTS ts,PetscTSRKType*)
    PetscErrorCode TSARKIMEXSetType(PetscTS ts,PetscTSRKType)
    PetscErrorCode TSARKIMEXSetFullyImplicit(PetscTS ts,PetscBool)

    PetscErrorCode TSPythonSetType(PetscTS,char[])
    PetscErrorCode TSPythonGetType(PetscTS,char*[])

cdef extern from * nogil:
    struct _p_TSAdapt
    ctypedef _p_TSAdapt *PetscTSAdapt "TSAdapt"
    PetscErrorCode TSGetAdapt(PetscTS,PetscTSAdapt*)
    PetscErrorCode TSAdaptGetStepLimits(PetscTSAdapt,PetscReal*,PetscReal*)
    PetscErrorCode TSAdaptSetStepLimits(PetscTSAdapt,PetscReal,PetscReal)
    PetscErrorCode TSAdaptCheckStage(PetscTSAdapt,PetscTS,PetscReal,PetscVec,PetscBool*)

cdef extern from * nogil: # custom.h
    PetscErrorCode TSSetTimeStepNumber(PetscTS,PetscInt)

# -----------------------------------------------------------------------------

cdef inline TS ref_TS(PetscTS ts):
    cdef TS ob = <TS> TS()
    ob.ts = ts
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

cdef PetscErrorCode TS_RHSFunction(
    PetscTS   ts,
    PetscReal t,
    PetscVec  x,
    PetscVec  f,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts   = ref_TS(ts)
    cdef Vec Xvec = ref_Vec(x)
    cdef Vec Fvec = ref_Vec(f)
    cdef object context = Ts.get_attr('__rhsfunction__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (function, args, kargs) = context
    function(Ts, toReal(t), Xvec, Fvec, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TS_RHSJacobian(
    PetscTS   ts,
    PetscReal t,
    PetscVec  x,
    PetscMat  J,
    PetscMat  P,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts   = ref_TS(ts)
    cdef Vec Xvec = ref_Vec(x)
    cdef Mat Jmat = ref_Mat(J)
    cdef Mat Pmat = ref_Mat(P)
    cdef object context = Ts.get_attr('__rhsjacobian__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(Ts, toReal(t), Xvec, Jmat, Pmat, *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode TS_IFunction(
    PetscTS   ts,
    PetscReal t,
    PetscVec  x,
    PetscVec  xdot,
    PetscVec  f,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts    = ref_TS(ts)
    cdef Vec Xvec  = ref_Vec(x)
    cdef Vec XDvec = ref_Vec(xdot)
    cdef Vec Fvec  = ref_Vec(f)
    cdef object context = Ts.get_attr('__ifunction__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (function, args, kargs) = context
    function(Ts, toReal(t), Xvec, XDvec, Fvec, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TS_IJacobian(
    PetscTS   ts,
    PetscReal t,
    PetscVec  x,
    PetscVec  xdot,
    PetscReal a,
    PetscMat  J,
    PetscMat  P,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS   Ts    = ref_TS(ts)
    cdef Vec  Xvec  = ref_Vec(x)
    cdef Vec  XDvec = ref_Vec(xdot)
    cdef Mat  Jmat  = ref_Mat(J)
    cdef Mat  Pmat  = ref_Mat(P)
    cdef object context = Ts.get_attr('__ijacobian__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(Ts, toReal(t), Xvec, XDvec, toReal(a), Jmat, Pmat, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TS_IJacobianP(
    PetscTS   ts,
    PetscReal t,
    PetscVec  x,
    PetscVec  xdot,
    PetscReal a,
    PetscMat  J,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS   Ts    = ref_TS(ts)
    cdef Vec  Xvec  = ref_Vec(x)
    cdef Vec  XDvec = ref_Vec(xdot)
    cdef Mat  Jmat  = ref_Mat(J)
    cdef object context = Ts.get_attr('__ijacobianp__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(Ts, toReal(t), Xvec, XDvec, toReal(a), Jmat, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TS_I2Function(
    PetscTS   ts,
    PetscReal t,
    PetscVec  x,
    PetscVec  xdot,
    PetscVec  xdotdot,
    PetscVec  f,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts    = ref_TS(ts)
    cdef Vec Xvec  = ref_Vec(x)
    cdef Vec XDvec = ref_Vec(xdot)
    cdef Vec XDDvec = ref_Vec(xdotdot)
    cdef Vec Fvec  = ref_Vec(f)
    cdef object context = Ts.get_attr('__i2function__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (function, args, kargs) = context
    function(Ts, toReal(t), Xvec, XDvec, XDDvec, Fvec, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TS_I2Jacobian(
    PetscTS   ts,
    PetscReal t,
    PetscVec  x,
    PetscVec  xdot,
    PetscVec  xdotdot,
    PetscReal v,
    PetscReal a,
    PetscMat  J,
    PetscMat  P,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS   Ts    = ref_TS(ts)
    cdef Vec  Xvec  = ref_Vec(x)
    cdef Vec  XDvec = ref_Vec(xdot)
    cdef Vec  XDDvec = ref_Vec(xdotdot)
    cdef Mat  Jmat  = ref_Mat(J)
    cdef Mat  Pmat  = ref_Mat(P)
    cdef object context = Ts.get_attr('__i2jacobian__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(Ts, toReal(t), Xvec, XDvec, XDDvec, toReal(v), toReal(a), Jmat, Pmat, *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode TS_Monitor(
    PetscTS   ts,
    PetscInt  step,
    PetscReal time,
    PetscVec  u,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts = ref_TS(ts)
    cdef Vec Vu = ref_Vec(u)
    cdef object monitorlist = Ts.get_attr('__monitor__')
    if monitorlist is None: return PETSC_SUCCESS
    for (monitor, args, kargs) in monitorlist:
        monitor(Ts, toInt(step), toReal(time), Vu, *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode TS_EventHandler(
    PetscTS     ts,
    PetscReal   time,
    PetscVec    u,
    PetscScalar fvalue[],
    void*       ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts = ref_TS(ts)
    cdef Vec Vu = ref_Vec(u)
    cdef object context = Ts.get_attr('__eventhandler__')
    if context is None: return PETSC_SUCCESS
    (eventhandler, args, kargs) = context
    cdef PetscInt nevents = 0
    CHKERR( TSGetNumEvents(ts, &nevents) )
    cdef npy_intp s = <npy_intp> nevents
    fvalue_array = PyArray_SimpleNewFromData(1, &s, NPY_PETSC_SCALAR, fvalue)
    eventhandler(Ts, toReal(time), Vu, fvalue_array, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TS_PostEvent(
    PetscTS   ts,
    PetscInt  nevents_zero,
    PetscInt  events_zero[],
    PetscReal time,
    PetscVec  u,
    PetscBool forward,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts = ref_TS(ts)
    cdef Vec Vu = ref_Vec(u)
    cdef object context = Ts.get_attr('__postevent__')
    if context is None: return PETSC_SUCCESS
    (postevent, args, kargs) = context
    cdef npy_intp s = <npy_intp> nevents_zero
    events_zero_array = PyArray_SimpleNewFromData(1, &s, NPY_PETSC_INT, events_zero)
    postevent(Ts, events_zero_array, toReal(time), Vu, toBool(forward), *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TS_PreStep(
    PetscTS ts,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS Ts = ref_TS(ts)
    (prestep, args, kargs) = Ts.get_attr('__prestep__')
    prestep(Ts, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode TS_PostStep(
    PetscTS ts,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS Ts = ref_TS(ts)
    (poststep, args, kargs) = Ts.get_attr('__poststep__')
    poststep(Ts, *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------

cdef PetscErrorCode TS_RHSJacobianP(
    PetscTS   ts,
    PetscReal t,
    PetscVec  x,
    PetscMat  J,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts   = ref_TS(ts)
    cdef Vec Xvec = ref_Vec(x)
    cdef Mat Jmat = ref_Mat(J)
    cdef object context = Ts.get_attr('__rhsjacobianp__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobianp, args, kargs) = context
    jacobianp(Ts, toReal(t), Xvec, Jmat, *args, **kargs)
    return PETSC_SUCCESS

# -----------------------------------------------------------------------------
