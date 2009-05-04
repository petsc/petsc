cdef extern from "petscts.h" nogil:

    ctypedef char* PetscTSType "const char*"
    PetscTSType TS_EULER
    PetscTSType TS_RUNGE_KUTTA
    PetscTSType TS_BEULER
    PetscTSType TS_CRANK_NICHOLSON
    PetscTSType TS_PSEUDO
    PetscTSType TS_SUNDIALS

    ctypedef enum PetscTSProblemType "TSProblemType":
        TS_LINEAR
        TS_NONLINEAR

    ctypedef int PetscTSCtxDel(void*)

    ctypedef int PetscTSFunction(PetscTS,
                                 PetscReal,
                                 PetscVec,
                                 PetscVec,
                                 void*) except PETSC_ERR_PYTHON

    ctypedef int PetscTSJacobian(PetscTS,
                                 PetscReal,
                                 PetscVec,
                                 PetscMat*,
                                 PetscMat*,
                                 PetscMatStructure*,
                                 void*) except PETSC_ERR_PYTHON

    ctypedef int PetscTSMonitor(PetscTS,
                                PetscInt,
                                PetscReal,
                                PetscVec,
                                void*) except PETSC_ERR_PYTHON

    int TSCreate(MPI_Comm comm, PetscTS*)
    int TSDestroy(PetscTS)
    int TSView(PetscTS,PetscViewer OPTIONAL)

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
    int TSSetRHSFunction(PetscTS,PetscVec,PetscTSFunction*,void*)
    int TSGetRHSFunction(PetscTS,PetscVec*,PetscTSFunction**,void*)
    int TSSetRHSJacobian(PetscTS,PetscMat,PetscMat,PetscTSJacobian*,void*)
    int TSGetRHSJacobian(PetscTS,PetscMat*,PetscMat*,PetscTSJacobian**,void**)
    int TSGetKSP(PetscTS,PetscKSP*)
    int TSGetSNES(PetscTS,PetscSNES*)

    int TSComputeRHSFunction(PetscTS,PetscReal,PetscVec,PetscVec)
    int TSComputeRHSJacobian(PetscTS,PetscReal,PetscVec,PetscMat*,PetscMat*,PetscMatStructure*)

    int TSSetTime(PetscTS,PetscReal)
    int TSGetTime(PetscTS,PetscReal*)
    int TSSetInitialTimeStep(PetscTS,PetscReal,PetscReal)
    int TSSetTimeStep(PetscTS,PetscReal)
    int TSGetTimeStep(PetscTS,PetscReal*)
    int TSSetTimeStepNumber(PetscTS,PetscInt)
    int TSGetTimeStepNumber(PetscTS,PetscInt*)
    int TSSetDuration(PetscTS,PetscInt,PetscReal)
    int TSGetDuration(PetscTS,PetscInt*,PetscReal*)

    int TSMonitorSet(PetscTS,PetscTSMonitor*,void*,PetscTSCtxDel*)
    int TSMonitorCancel(PetscTS)

    int TSSetUp(PetscTS)
    int TSStep(PetscTS,PetscInt*,PetscReal*)
    int TSSolve(PetscTS,PetscVec)

cdef extern from "custom.h" nogil:
    int TSSetUseFDColoring(PetscTS,PetscTruth)
    int TSGetUseFDColoring(PetscTS,PetscTruth*)

# --------------------------------------------------------------------

cdef inline TS ref_TS(PetscTS ts):
    cdef TS ob = <TS> TS()
    PetscIncref(<PetscObject>ts)
    ob.ts = ts
    return ob

# --------------------------------------------------------------------

cdef inline object TS_getFun(PetscTS ts):
    return Object_getAttr(<PetscObject>ts, '__function__')

cdef int TS_Function(PetscTS ts,
                     PetscReal t,
                     PetscVec  x,
                     PetscVec  f,
                     void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts   = ref_TS(ts)
    cdef Vec Xvec = ref_Vec(x)
    cdef Vec Fvec = ref_Vec(f)
    (function, args, kargs) = TS_getFun(ts)
    function(Ts, toReal(t), Xvec, Fvec, *args, **kargs)
    return 0

cdef inline int TS_setFun(PetscTS ts, PetscVec f, object fun) except -1:
    CHKERR( TSSetRHSFunction(ts, f, TS_Function, NULL) )
    Object_setAttr(<PetscObject>ts, '__function__', fun)
    return 0

# --------------------------------------------------------------------

cdef inline object TS_getJac(PetscTS ts):
    return Object_getAttr(<PetscObject>ts, '__jacobian__')

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
    (jacobian, args, kargs) = TS_getJac(ts)
    retv = jacobian(Ts, toReal(t), Xvec, Jmat, Pmat, *args, **kargs)
    s[0] = matstructure(retv)
    cdef PetscMat Jtmp = NULL, Ptmp = NULL
    Jtmp = J[0]; J[0] = Jmat.mat; Jmat.mat = Jtmp
    Ptmp = P[0]; P[0] = Pmat.mat; Pmat.mat = Ptmp
    return 0

cdef inline int TS_setJac(PetscTS ts,
                          PetscMat J, PetscMat P,
                          object jac) except -1:
    CHKERR( TSSetRHSJacobian(ts, J, P, TS_Jacobian, NULL) )
    Object_setAttr(<PetscObject>ts, '__jacobian__', jac)
    return 0

# --------------------------------------------------------------------

cdef inline object TS_getMon(PetscTS ts):
    return Object_getAttr(<PetscObject>ts, '__monitor__')

cdef int TS_Monitor(PetscTS    ts,
                    PetscInt   step,
                    PetscReal  time,
                    PetscVec   u,
                    void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef object monitorlist = TS_getMon(ts)
    if monitorlist is None: return 0
    cdef TS  Ts = ref_TS(ts)
    cdef Vec Vu = ref_Vec(u)
    for (monitor, args, kargs) in monitorlist:
        monitor(Ts, step, toReal(time), Vu, *args, **kargs)
    return 0

cdef inline int TS_setMon(PetscTS ts, object mon) except -1:
    if mon is None: return 0
    CHKERR( TSMonitorSet(ts, TS_Monitor, NULL, NULL) )
    cdef object monitorlist = TS_getMon(ts)
    if monitorlist is None: monitorlist = [mon]
    else: monitorlist.append(mon)
    Object_setAttr(<PetscObject>ts, '__monitor__', monitorlist)
    return 0

cdef inline int TS_clsMon(PetscTS ts) except -1:
    CHKERR( TSMonitorCancel(ts) )
    Object_setAttr(<PetscObject>ts, '__monitor__', None)
    return 0

# --------------------------------------------------------------------

cdef extern from "libpetsc4py.h":
    PetscTSType TS_PYTHON
    int TSPythonSetContext(PetscTS,void*)
    int TSPythonGetContext(PetscTS,void**)
    int TSPythonSetType(PetscTS,char[])

# --------------------------------------------------------------------
