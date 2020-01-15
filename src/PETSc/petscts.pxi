cdef extern from * nogil:

    ctypedef char* PetscTSType "const char*"
    PetscTSType TSEULER
    PetscTSType TSBEULER
    PetscTSType TSBASICSYMPLECTIC
    PetscTSType TSPSEUDO
    PetscTSType TSCN
    PetscTSType TSSUNDIALS
    PetscTSType TSRK
    #PetscTSType TSPYTHON
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

    ctypedef int PetscTSCtxDel(void*)

    ctypedef int (*PetscTSFunctionFunction)(PetscTS,
                                            PetscReal,
                                            PetscVec,
                                            PetscVec,
                                            void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscTSJacobianFunction)(PetscTS,
                                            PetscReal,
                                            PetscVec,
                                            PetscMat,
                                            PetscMat,
                                            void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscTSIFunctionFunction)(PetscTS,
                                             PetscReal,
                                             PetscVec,
                                             PetscVec,
                                             PetscVec,
                                             void*) except PETSC_ERR_PYTHON
    ctypedef int (*PetscTSIJacobianFunction)(PetscTS,
                                             PetscReal,
                                             PetscVec,
                                             PetscVec,
                                             PetscReal,
                                             PetscMat,
                                             PetscMat,
                                             void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscTSI2FunctionFunction)(PetscTS,
                                              PetscReal,
                                              PetscVec,
                                              PetscVec,
                                              PetscVec,
                                              PetscVec,
                                              void*) except PETSC_ERR_PYTHON
    ctypedef int (*PetscTSI2JacobianFunction)(PetscTS,
                                              PetscReal,
                                              PetscVec,
                                              PetscVec,
                                              PetscVec,
                                              PetscReal,
                                              PetscReal,
                                              PetscMat,
                                              PetscMat,
                                              void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscTSMonitorFunction)(PetscTS,
                                           PetscInt,
                                           PetscReal,
                                           PetscVec,
                                           void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscTSPreStepFunction)  (PetscTS) except PETSC_ERR_PYTHON
    ctypedef int (*PetscTSPostStepFunction) (PetscTS) except PETSC_ERR_PYTHON

    int TSCreate(MPI_Comm comm,PetscTS*)
    int TSClone(PetscTS,PetscTS*)
    int TSDestroy(PetscTS*)
    int TSView(PetscTS,PetscViewer)
    int TSLoad(PetscTS,PetscViewer)

    int TSSetProblemType(PetscTS,PetscTSProblemType)
    int TSGetProblemType(PetscTS,PetscTSProblemType*)
    int TSSetEquationType(PetscTS,PetscTSEquationType)
    int TSGetEquationType(PetscTS,PetscTSEquationType*)
    int TSSetType(PetscTS,PetscTSType)
    int TSGetType(PetscTS,PetscTSType*)

    int TSSetOptionsPrefix(PetscTS,char[])
    int TSAppendOptionsPrefix(PetscTS,char[])
    int TSGetOptionsPrefix(PetscTS,char*[])
    int TSSetFromOptions(PetscTS)

    int TSSetSolution(PetscTS,PetscVec)
    int TSGetSolution(PetscTS,PetscVec*)
    int TS2SetSolution(PetscTS,PetscVec,PetscVec)
    int TS2GetSolution(PetscTS,PetscVec*,PetscVec*)

    int TSGetRHSFunction(PetscTS,PetscVec*,PetscTSFunctionFunction*,void*)
    int TSGetRHSJacobian(PetscTS,PetscMat*,PetscMat*,PetscTSJacobianFunction*,void**)
    int TSSetRHSFunction(PetscTS,PetscVec,PetscTSFunctionFunction,void*)
    int TSSetRHSJacobian(PetscTS,PetscMat,PetscMat,PetscTSJacobianFunction,void*)
    int TSSetIFunction(PetscTS,PetscVec,PetscTSIFunctionFunction,void*)
    int TSSetIJacobian(PetscTS,PetscMat,PetscMat,PetscTSIJacobianFunction,void*)
    int TSGetIFunction(PetscTS,PetscVec*,PetscTSIFunctionFunction*,void*)
    int TSGetIJacobian(PetscTS,PetscMat*,PetscMat*,PetscTSIJacobianFunction*,void**)
    int TSSetI2Function(PetscTS,PetscVec,PetscTSI2FunctionFunction,void*)
    int TSSetI2Jacobian(PetscTS,PetscMat,PetscMat,PetscTSI2JacobianFunction,void*)
    int TSGetI2Function(PetscTS,PetscVec*,PetscTSI2FunctionFunction*,void**)
    int TSGetI2Jacobian(PetscTS,PetscMat*,PetscMat*,PetscTSI2JacobianFunction*,void**)

    int TSGetKSP(PetscTS,PetscKSP*)
    int TSGetSNES(PetscTS,PetscSNES*)

    int TSGetDM(PetscTS,PetscDM*)
    int TSSetDM(PetscTS,PetscDM)

    int TSComputeRHSFunction(PetscTS,PetscReal,PetscVec,PetscVec)
    int TSComputeRHSFunctionLinear(PetscTS,PetscReal,PetscVec,PetscVec,void*)
    int TSComputeRHSJacobian(PetscTS,PetscReal,PetscVec,PetscMat,PetscMat)
    int TSComputeRHSJacobianConstant(PetscTS,PetscReal,PetscVec,PetscMat,PetscMat,void*)
    int TSComputeIFunction(PetscTS,PetscReal,PetscVec,PetscVec,PetscVec,PetscBool)
    int TSComputeIJacobian(PetscTS,PetscReal,PetscVec,PetscVec,PetscReal,PetscMat,PetscMat,PetscBool)
    int TSComputeI2Function(PetscTS,PetscReal,PetscVec,PetscVec,PetscVec,PetscVec)
    int TSComputeI2Jacobian(PetscTS,PetscReal,PetscVec,PetscVec,PetscVec,PetscReal,PetscReal,PetscMat,PetscMat)

    int TSSetTime(PetscTS,PetscReal)
    int TSGetTime(PetscTS,PetscReal*)
    int TSGetPrevTime(PetscTS,PetscReal*)
    int TSGetSolveTime(PetscTS,PetscReal*)
    int TSSetTimeStep(PetscTS,PetscReal)
    int TSGetTimeStep(PetscTS,PetscReal*)
    int TSSetStepNumber(PetscTS,PetscInt)
    int TSGetStepNumber(PetscTS,PetscInt*)
    int TSSetMaxSteps(PetscTS,PetscInt)
    int TSGetMaxSteps(PetscTS,PetscInt*)
    int TSSetMaxTime(PetscTS,PetscReal)
    int TSGetMaxTime(PetscTS,PetscReal*)
    int TSSetExactFinalTime(PetscTS,PetscTSExactFinalTimeOption)
    int TSSetConvergedReason(PetscTS,PetscTSConvergedReason)
    int TSGetConvergedReason(PetscTS,PetscTSConvergedReason*)
    int TSGetSNESIterations(PetscTS,PetscInt*)
    int TSGetKSPIterations(PetscTS,PetscInt*)
    int TSGetStepRejections(PetscTS,PetscInt*)
    int TSSetMaxStepRejections(PetscTS,PetscInt)
    int TSGetSNESFailures(PetscTS,PetscInt*)
    int TSSetMaxSNESFailures(PetscTS,PetscInt)
    int TSSetErrorIfStepFails(PetscTS,PetscBool)
    int TSSetTolerances(PetscTS,PetscReal,PetscVec,PetscReal,PetscVec)
    int TSGetTolerances(PetscTS,PetscReal*,PetscVec*,PetscReal*,PetscVec*)

    int TSMonitorSet(PetscTS,PetscTSMonitorFunction,void*,PetscTSCtxDel*)
    int TSMonitorCancel(PetscTS)
    int TSMonitor(PetscTS,PetscInt,PetscReal,PetscVec)

    ctypedef int (*PetscTSAdjointR)(PetscTS,PetscReal,PetscVec,PetscVec,void*) except PETSC_ERR_PYTHON
    ctypedef int (*PetscTSAdjointDRDY)(PetscTS,PetscReal,PetscVec,PetscVec[],void*) except PETSC_ERR_PYTHON
    ctypedef int (*PetscTSAdjointDRDP)(PetscTS,PetscReal,PetscVec,PetscVec[],void*) except PETSC_ERR_PYTHON
    ctypedef int (*PetscTSRHSJacobianP)(PetscTS,PetscReal,PetscVec,PetscMat,void*) except PETSC_ERR_PYTHON

    int TSSetSaveTrajectory(PetscTS)
    int TSSetCostGradients(PetscTS,PetscInt,PetscVec*,PetscVec*)
    int TSGetCostGradients(PetscTS,PetscInt*,PetscVec**,PetscVec**)
    int TSCreateQuadratureTS(PetscTS,PetscBool,PetscTS*)
    int TSGetQuadratureTS(PetscTS,PetscBool*,PetscTS*)
    int TSGetCostIntegral(PetscTS,PetscVec*)

    int TSSetRHSJacobianP(PetscTS,PetscMat,PetscTSRHSJacobianP,void*)
    int TSComputeRHSJacobianP(PetscTS,PetscReal,PetscVec,PetscMat)

    int TSAdjointSolve(PetscTS)
    int TSAdjointSetSteps(PetscTS,PetscInt)
    int TSAdjointStep(PetscTS)
    int TSAdjointSetUp(PetscTS)
    int TSAdjointComputeDRDPFunction(PetscTS,PetscReal,PetscVec,PetscVec*)
    int TSAdjointComputeDRDYFunction(PetscTS,PetscReal,PetscVec,PetscVec*)
    int TSAdjointCostIntegral(PetscTS)

    int TSForwardSetSensitivities(PetscTS,PetscInt,PetscVec*,PetscInt,PetscVec*)
    int TSForwardGetSensitivities(PetscTS,PetscInt*,PetscVec**,PetscInt*,PetscVec**)
    int TSForwardSetIntegralGradients(PetscTS,PetscInt,PetscVec *,PetscVec *)
    int TSForwardGetIntegralGradients(PetscTS,PetscInt*,PetscVec **,PetscVec **)
    int TSForwardSetRHSJacobianP(PetscTS,PetscVec*,PetscTSCostIntegrandFunction,void*)
    int TSForwardComputeRHSJacobianP(PetscTS,PetscReal,PetscVec,PetscVec*)
    int TSForwardSetUp(PetscTS)
    int TSForwardCostIntegral(PetscTS)
    int TSForwardStep(PetscTS)

    int TSSetPreStep(PetscTS, PetscTSPreStepFunction)
    int TSSetPostStep(PetscTS, PetscTSPostStepFunction)

    int TSSetUp(PetscTS)
    int TSReset(PetscTS)
    int TSStep(PetscTS)
    int TSRestartStep(PetscTS)
    int TSRollBack(PetscTS)
    int TSSolve(PetscTS,PetscVec)
    int TSInterpolate(PetscTS,PetscReal,PetscVec)

    int TSThetaSetTheta(PetscTS,PetscReal)
    int TSThetaGetTheta(PetscTS,PetscReal*)
    int TSThetaSetEndpoint(PetscTS,PetscBool)
    int TSThetaGetEndpoint(PetscTS,PetscBool*)

    int TSAlphaSetRadius(PetscTS,PetscReal)
    int TSAlphaSetParams(PetscTS,PetscReal,PetscReal,PetscReal)
    int TSAlphaGetParams(PetscTS,PetscReal*,PetscReal*,PetscReal*)

    ctypedef char* PetscTSRKType "const char*"
    PetscTSRKType TSRK1FE
    PetscTSRKType TSRK2A
    PetscTSRKType TSRK3
    PetscTSRKType TSRK3BS
    PetscTSRKType TSRK4
    PetscTSRKType TSRK5F
    PetscTSRKType TSRK5DP
    PetscTSRKType TSRK5BS

    int TSRKGetType(PetscTS ts,PetscTSRKType*)
    int TSRKSetType(PetscTS ts,PetscTSRKType)

    ctypedef char* PetscTSARKIMEXType "const char*"
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

    int TSARKIMEXGetType(PetscTS ts,PetscTSRKType*)
    int TSARKIMEXSetType(PetscTS ts,PetscTSRKType)

cdef extern from "custom.h" nogil:
    int TSSetTimeStepNumber(PetscTS,PetscInt)

cdef extern from "libpetsc4py.h":
    PetscTSType TSPYTHON
    int TSPythonSetContext(PetscTS,void*)
    int TSPythonGetContext(PetscTS,void**)
    int TSPythonSetType(PetscTS,char[])

# -----------------------------------------------------------------------------

cdef inline TS ref_TS(PetscTS ts):
    cdef TS ob = <TS> TS()
    ob.ts = ts
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

cdef int TS_RHSFunction(
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
    return 0

cdef int TS_RHSJacobian(
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
    return 0

# -----------------------------------------------------------------------------

cdef int TS_IFunction(
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
    return 0

cdef int TS_IJacobian(
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
    return 0

cdef int TS_I2Function(
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
    return 0

cdef int TS_I2Jacobian(
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
    return 0

# -----------------------------------------------------------------------------

cdef int TS_Monitor(
    PetscTS   ts,
    PetscInt  step,
    PetscReal time,
    PetscVec  u,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts = ref_TS(ts)
    cdef Vec Vu = ref_Vec(u)
    cdef object monitorlist = Ts.get_attr('__monitor__')
    if monitorlist is None: return 0
    for (monitor, args, kargs) in monitorlist:
        monitor(Ts, toInt(step), toReal(time), Vu, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef int TS_PreStep(
    PetscTS ts,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS Ts = ref_TS(ts)
    (prestep, args, kargs) = Ts.get_attr('__prestep__')
    prestep(Ts, *args, **kargs)
    return 0

cdef int TS_PostStep(
    PetscTS ts,
    ) except PETSC_ERR_PYTHON with gil:
    cdef TS Ts = ref_TS(ts)
    (poststep, args, kargs) = Ts.get_attr('__poststep__')
    poststep(Ts, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef int TS_RHSJacobianP(
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
    return 0

# -----------------------------------------------------------------------------
