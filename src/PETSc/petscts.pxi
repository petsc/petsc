cdef extern from "petscts.h" nogil:

    ctypedef char* PetscTSType "const char*"
    PetscTSType TSEULER
    PetscTSType TSBEULER
    PetscTSType TSPSEUDO
    PetscTSType TSCRANK_NICHOLSON
    PetscTSType TSSUNDIALS
    PetscTSType TSRUNGE_KUTTA
    PetscTSType TSTHETA
    PetscTSType TSGL
    PetscTSType TSSSP

    ctypedef enum PetscTSProblemType "TSProblemType":
        TS_LINEAR
        TS_NONLINEAR

    ctypedef int PetscTSCtxDel(void*)

    ctypedef int (*PetscTSFunctionFunction)(PetscTS,
                                            PetscReal,
                                            PetscVec,
                                            PetscVec,
                                            void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscTSJacobianFunction)(PetscTS,
                                            PetscReal,
                                            PetscVec,
                                            PetscMat*,
                                            PetscMat*,
                                            PetscMatStructure*,
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
                                             PetscMat*,
                                             PetscMat*,
                                             PetscMatStructure*,
                                             void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscTSMonitorFunction)(PetscTS,
                                           PetscInt,
                                           PetscReal,
                                           PetscVec,
                                           void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscTSPreStepFunction)  (PetscTS) except PETSC_ERR_PYTHON
    ctypedef int (*PetscTSPostStepFunction) (PetscTS) except PETSC_ERR_PYTHON

    int TSCreate(MPI_Comm comm,PetscTS*)
    int TSDestroy(PetscTS)
    int TSView(PetscTS,PetscViewer)

    int TSSetProblemType(PetscTS,PetscTSProblemType)
    int TSGetProblemType(PetscTS,PetscTSProblemType*)
    int TSSetType(PetscTS,PetscTSType)
    int TSGetType(PetscTS,PetscTSType*)

    int TSSetOptionsPrefix(PetscTS,char[])
    int TSAppendOptionsPrefix(PetscTS,char[])
    int TSGetOptionsPrefix(PetscTS,char*[])
    int TSSetFromOptions(PetscTS)

    int TSSetSolution(PetscTS,PetscVec)
    int TSGetSolution(PetscTS,PetscVec*)
    int TSGetRHSFunction(PetscTS,PetscVec*,PetscTSFunctionFunction*,void*)
    int TSGetRHSJacobian(PetscTS,PetscMat*,PetscMat*,PetscTSJacobianFunction*,void**)
    int TSSetRHSFunction(PetscTS,PetscVec,PetscTSFunctionFunction,void*)
    int TSSetRHSJacobian(PetscTS,PetscMat,PetscMat,PetscTSJacobianFunction,void*)
    int TSSetIFunction(PetscTS,PetscTSIFunctionFunction,void*)
    int TSSetIJacobian(PetscTS,PetscMat,PetscMat,PetscTSIJacobianFunction,void*)

    int TSGetKSP(PetscTS,PetscKSP*)
    int TSGetSNES(PetscTS,PetscSNES*)

    int TSComputeRHSFunction(PetscTS,PetscReal,PetscVec,PetscVec)
    int TSComputeRHSJacobian(PetscTS,PetscReal,PetscVec,PetscMat*,PetscMat*,PetscMatStructure*)
    int TSComputeIFunction(PetscTS,PetscReal,PetscVec,PetscVec,PetscVec,)
    int TSComputeIJacobian(PetscTS,PetscReal,PetscVec,PetscVec,PetscReal,PetscMat*,PetscMat*,PetscMatStructure*)

    int TSSetTime(PetscTS,PetscReal)
    int TSGetTime(PetscTS,PetscReal*)
    int TSSetInitialTimeStep(PetscTS,PetscReal,PetscReal)
    int TSSetTimeStep(PetscTS,PetscReal)
    int TSGetTimeStep(PetscTS,PetscReal*)
    int TSSetTimeStepNumber(PetscTS,PetscInt)
    int TSGetTimeStepNumber(PetscTS,PetscInt*)
    int TSSetDuration(PetscTS,PetscInt,PetscReal)
    int TSGetDuration(PetscTS,PetscInt*,PetscReal*)

    int TSMonitorSet(PetscTS,PetscTSMonitorFunction,void*,PetscTSCtxDel*)
    int TSMonitorCancel(PetscTS)

    int TSSetPreStep(PetscTS, PetscTSPreStepFunction)
    int TSSetPostStep(PetscTS, PetscTSPostStepFunction)

    int TSSetUp(PetscTS)
    int TSStep(PetscTS,PetscInt*,PetscReal*)
    int TSSolve(PetscTS,PetscVec)

cdef extern from "custom.h" nogil:
    int TSSetUseFDColoring(PetscTS,PetscTruth)
    int TSGetUseFDColoring(PetscTS,PetscTruth*)
    int TSMonitorCall(PetscTS,PetscInt,PetscReal,PetscVec)

# -----------------------------------------------------------------------------

cdef inline TS ref_TS(PetscTS ts):
    cdef TS ob = <TS> TS()
    PetscIncref(<PetscObject>ts)
    ob.ts = ts
    return ob

# -----------------------------------------------------------------------------

cdef inline object TS_getFunction(PetscTS ts):
    return Object_getAttr(<PetscObject>ts, '__function__')

cdef inline int TS_setFunction(PetscTS ts, PetscVec f, object fun) except -1:
    CHKERR( TSSetRHSFunction(ts, f, TS_Function, NULL) )
    Object_setAttr(<PetscObject>ts, '__function__', fun)
    return 0

cdef int TS_Function(PetscTS ts,
                     PetscReal t,
                     PetscVec  x,
                     PetscVec  f,
                     void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts   = ref_TS(ts)
    cdef Vec Xvec = ref_Vec(x)
    cdef Vec Fvec = ref_Vec(f)
    (function, args, kargs) = TS_getFunction(ts)
    function(Ts, toReal(t), Xvec, Fvec, *args, **kargs)
    return 0

cdef inline object TS_getJacobian(PetscTS ts):
    return Object_getAttr(<PetscObject>ts, '__jacobian__')

cdef inline int TS_setJacobian(PetscTS ts,
                               PetscMat J, PetscMat P,
                               object jacobian) except -1:
    CHKERR( TSSetRHSJacobian(ts, J, P, TS_Jacobian, NULL) )
    Object_setAttr(<PetscObject>ts, '__jacobian__', jacobian)
    return 0

cdef int TS_Jacobian(PetscTS ts,
                     PetscReal t,
                     PetscVec  x,
                     PetscMat  *J,
                     PetscMat  *P,
                     PetscMatStructure* s,
                     void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TS   Ts   = ref_TS(ts)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Mat  Jmat = ref_Mat(J[0])
    cdef Mat  Pmat = ref_Mat(P[0])
    (jacobian, args, kargs) = TS_getJacobian(ts)
    retv = jacobian(Ts, toReal(t), Xvec, Jmat, Pmat, *args, **kargs)
    s[0] = matstructure(retv)
    cdef PetscMat Jtmp = NULL, Ptmp = NULL
    Jtmp = J[0]; J[0] = Jmat.mat; Jmat.mat = Jtmp
    Ptmp = P[0]; P[0] = Pmat.mat; Pmat.mat = Ptmp
    return 0

# -----------------------------------------------------------------------------

cdef inline object TS_getIFunction(PetscTS ts):
    return Object_getAttr(<PetscObject>ts, '__function__')

cdef inline int TS_setIFunction(PetscTS ts, PetscVec f, 
                                object function) except -1:
    CHKERR( TSSetIFunction(ts, TS_IFunction, NULL) )
    CHKERR( PetscObjectCompose(<PetscObject>ts, 
                                "__i_funcvec__", <PetscObject>f) )
    Object_setAttr(<PetscObject>ts, '__function__', function)
    return 0

cdef int TS_IFunction(PetscTS ts,
                      PetscReal t,
                      PetscVec  x,
                      PetscVec  xdot,
                      PetscVec  f,
                      void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts    = ref_TS(ts)
    cdef Vec Xvec  = ref_Vec(x)
    cdef Vec XDvec = ref_Vec(xdot)
    cdef Vec Fvec  = ref_Vec(f)
    (function, args, kargs) = TS_getIFunction(ts)
    function(Ts, toReal(t), Xvec, XDvec, Fvec, *args, **kargs)
    return 0

cdef inline object TS_getIJacobian(PetscTS ts):
    return Object_getAttr(<PetscObject>ts, '__jacobian__')

cdef inline int TS_setIJacobian(PetscTS ts,
                                PetscMat J, PetscMat P,
                                object jacobian) except -1:
    CHKERR( TSSetIJacobian(ts, J, P, TS_IJacobian, NULL) )
    CHKERR( PetscObjectCompose(<PetscObject>ts,
                                "__i_pjacmat__", <PetscObject>P) )
    Object_setAttr(<PetscObject>ts, '__jacobian__', jacobian)
    return 0

cdef int TS_IJacobian(PetscTS ts,
                      PetscReal t,
                      PetscVec  x,
                      PetscVec  xdot,
                      PetscReal a,
                      PetscMat  *J,
                      PetscMat  *P,
                      PetscMatStructure* s,
                      void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TS   Ts    = ref_TS(ts)
    cdef Vec  Xvec  = ref_Vec(x)
    cdef Vec  XDvec = ref_Vec(xdot)
    cdef Mat  Jmat  = ref_Mat(J[0])
    cdef Mat  Pmat  = ref_Mat(P[0])
    (jacobian, args, kargs) = TS_getJacobian(ts)
    retv = jacobian(Ts, toReal(t), Xvec, XDvec, toReal(a), 
                    Jmat, Pmat, *args, **kargs)
    s[0] = matstructure(retv)
    cdef PetscMat Jtmp = NULL, Ptmp = NULL
    Jtmp = J[0]; J[0] = Jmat.mat; Jmat.mat = Jtmp
    Ptmp = P[0]; P[0] = Pmat.mat; Pmat.mat = Ptmp
    return 0

# -----------------------------------------------------------------------------

cdef inline object TS_getMonitor(PetscTS ts):
    return Object_getAttr(<PetscObject>ts, '__monitor__')

cdef inline int TS_setMonitor(PetscTS ts, object monitor) except -1:
    CHKERR( TSMonitorSet(ts, TS_Monitor, NULL, NULL) )
    cdef object monitorlist = TS_getMonitor(ts)
    if monitor is None: monitorlist = None
    elif monitorlist is None: monitorlist = [monitor]
    else: monitorlist.append(monitor)
    Object_setAttr(<PetscObject>ts, '__monitor__', monitorlist)
    return 0

cdef inline int TS_delMonitor(PetscTS ts) except -1:
    Object_setAttr(<PetscObject>ts, '__monitor__', None)
    return 0

cdef int TS_Monitor(PetscTS    ts,
                    PetscInt   step,
                    PetscReal  time,
                    PetscVec   u,
                    void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef object monitorlist = TS_getMonitor(ts)
    if monitorlist is None: return 0
    cdef TS  Ts = ref_TS(ts)
    cdef Vec Vu = ref_Vec(u)
    for (monitor, args, kargs) in monitorlist:
        monitor(Ts, step, toReal(time), Vu, *args, **kargs)
    return 0

# --------------------------------------------------------------------

cdef inline object TS_getPreStep(PetscTS ts):
    return Object_getAttr(<PetscObject>ts, '__prestep__')

cdef int TS_PreStep(PetscTS ts) except PETSC_ERR_PYTHON with gil:
    cdef TS Ts = ref_TS(ts)
    (prestep, args, kargs) = TS_getPreStep(ts)
    prestep(Ts, *args, **kargs)
    return 0

cdef inline int TS_setPreStep(PetscTS ts, object prestep) except -1:
    if prestep is None: CHKERR( TSSetPreStep(ts, NULL) )
    else: CHKERR( TSSetPreStep(ts, TS_PreStep) )
    Object_setAttr(<PetscObject>ts, '__prestep__', prestep)
    return 0

# --

cdef inline object TS_getPostStep(PetscTS ts):
    return Object_getAttr(<PetscObject>ts, '__poststep__')

cdef inline int TS_setPostStep(PetscTS ts, object poststep) except -1:
    if poststep is None: CHKERR( TSSetPostStep(ts, NULL) )
    else: CHKERR( TSSetPostStep(ts, TS_PostStep) )
    Object_setAttr(<PetscObject>ts, '__poststep__', poststep)
    return 0

cdef int TS_PostStep(PetscTS ts) except PETSC_ERR_PYTHON with gil:
    cdef TS Ts = ref_TS(ts)
    (poststep, args, kargs) = TS_getPostStep(ts)
    poststep(Ts, *args, **kargs)
    return 0

# --------------------------------------------------------------------

cdef extern from "libpetsc4py.h":
    PetscTSType TSPYTHON
    int TSPythonSetContext(PetscTS,void*)
    int TSPythonGetContext(PetscTS,void**)
    int TSPythonSetType(PetscTS,char[])

# --------------------------------------------------------------------
