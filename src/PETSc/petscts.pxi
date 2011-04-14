cdef extern from * nogil:

    ctypedef char* PetscTSType "const char*"
    PetscTSType TSEULER
    PetscTSType TSBEULER
    PetscTSType TSPSEUDO
    PetscTSType TSCN
    PetscTSType TSSUNDIALS
    PetscTSType TSRK
    #PetscTSType TSPYTHON
    PetscTSType TSTHETA
    PetscTSType TSALPHA
    PetscTSType TSGL
    PetscTSType TSSSP

    ctypedef enum PetscTSProblemType "TSProblemType":
        TS_LINEAR
        TS_NONLINEAR

    ctypedef int PetscTSCtxDel(void*)

    ctypedef int (*PetscTSMatrixFunction)(PetscTS,
                                          PetscReal,
                                          PetscMat*,
                                          PetscMat*,
                                          PetscMatStructure*,
                                          void*) except PETSC_ERR_PYTHON

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

    int TSSetMatrices(PetscTS,PetscMat,PetscTSMatrixFunction,PetscMat,PetscTSMatrixFunction,PetscMatStructure,void*)

    int TSGetRHSFunction(PetscTS,PetscVec*,PetscTSFunctionFunction*,void*)
    int TSGetRHSJacobian(PetscTS,PetscMat*,PetscMat*,PetscTSJacobianFunction*,void**)
    int TSSetRHSFunction(PetscTS,PetscVec,PetscTSFunctionFunction,void*)
    int TSSetRHSJacobian(PetscTS,PetscMat,PetscMat,PetscTSJacobianFunction,void*)
    int TSSetIFunction(PetscTS,PetscVec,PetscTSIFunctionFunction,void*)
    int TSSetIJacobian(PetscTS,PetscMat,PetscMat,PetscTSIJacobianFunction,void*)
    int TSGetIJacobian(PetscTS,PetscMat*,PetscMat*,PetscTSIJacobianFunction*,void**)

    int TSGetKSP(PetscTS,PetscKSP*)
    int TSGetSNES(PetscTS,PetscSNES*)

    int TSGetDM(PetscTS,PetscDM*)
    int TSSetDM(PetscTS,PetscDM)

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
    int TSReset(PetscTS)
    int TSStep(PetscTS,PetscInt*,PetscReal*)
    int TSSolve(PetscTS,PetscVec)

    int TSThetaSetTheta(PetscTS,PetscReal)
    int TSThetaGetTheta(PetscTS,PetscReal*)

cdef extern from "custom.h" nogil:
    int TSSetUseFDColoring(PetscTS,PetscBool)
    int TSGetUseFDColoring(PetscTS,PetscBool*)
    int TSMonitorCall(PetscTS,PetscInt,PetscReal,PetscVec)

cdef extern from "libpetsc4py.h":
    PetscTSType TSPYTHON
    int TSPythonSetContext(PetscTS,void*)
    int TSPythonGetContext(PetscTS,void**)
    int TSPythonSetType(PetscTS,char[])

# -----------------------------------------------------------------------------

cdef inline TS ref_TS(PetscTS ts):
    cdef TS ob = <TS> TS()
    PetscIncref(<PetscObject>ts)
    ob.ts = ts
    return ob

# -----------------------------------------------------------------------------

cdef int TS_LHSMatrix(PetscTS ts,
                      PetscReal t,
                      PetscMat *A,
                      PetscMat *B,
                      PetscMatStructure* s,
                      void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TS   Ts   = ref_TS(ts)
    cdef Mat  Amat = ref_Mat(A[0])
    (lhsmatrix, args, kargs) = Ts.get_attr('__lhsmatrix__')
    retv = lhsmatrix(Ts, toReal(t), Amat, *args, **kargs)
    s[0] = matstructure(retv)
    cdef PetscMat Atmp = NULL
    Atmp = A[0]; A[0] = Amat.mat; Amat.mat = Atmp
    return 0

cdef int TS_RHSMatrix(PetscTS ts,
                      PetscReal t,
                      PetscMat *A,
                      PetscMat *B,
                      PetscMatStructure* s,
                      void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TS   Ts   = ref_TS(ts)
    cdef Mat  Amat = ref_Mat(A[0])
    (rhsmatrix, args, kargs) = Ts.get_attr('__rhsmatrix__')
    retv = rhsmatrix(Ts, toReal(t), Amat, *args, **kargs)
    s[0] = matstructure(retv)
    cdef PetscMat Atmp = NULL
    Atmp = A[0]; A[0] = Amat.mat; Amat.mat = Atmp
    return 0

# -----------------------------------------------------------------------------

cdef int TS_RHSFunction(PetscTS ts,
                        PetscReal t,
                        PetscVec  x,
                        PetscVec  f,
                        void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts   = ref_TS(ts)
    cdef Vec Xvec = ref_Vec(x)
    cdef Vec Fvec = ref_Vec(f)
    (function, args, kargs) = Ts.get_attr('__rhsfunction__')
    function(Ts, toReal(t), Xvec, Fvec, *args, **kargs)
    return 0

cdef int TS_RHSJacobian(PetscTS ts,
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
    (jacobian, args, kargs) = Ts.get_attr('__rhsjacobian__')
    retv = jacobian(Ts, toReal(t), Xvec, Jmat, Pmat, *args, **kargs)
    s[0] = matstructure(retv)
    cdef PetscMat Jtmp = NULL, Ptmp = NULL
    Jtmp = J[0]; J[0] = Jmat.mat; Jmat.mat = Jtmp
    Ptmp = P[0]; P[0] = Pmat.mat; Pmat.mat = Ptmp
    return 0

# -----------------------------------------------------------------------------

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
    (function, args, kargs) = Ts.get_attr('__ifunction__')
    function(Ts, toReal(t), Xvec, XDvec, Fvec, *args, **kargs)
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
    (jacobian, args, kargs) = Ts.get_attr('__ijacobian__')
    retv = jacobian(Ts, toReal(t), Xvec, XDvec, toReal(a),
                    Jmat, Pmat, *args, **kargs)
    s[0] = matstructure(retv)
    cdef PetscMat Jtmp = NULL, Ptmp = NULL
    Jtmp = J[0]; J[0] = Jmat.mat; Jmat.mat = Jtmp
    Ptmp = P[0]; P[0] = Pmat.mat; Pmat.mat = Ptmp
    return 0

# -----------------------------------------------------------------------------

cdef int TS_Monitor(PetscTS    ts,
                    PetscInt   step,
                    PetscReal  time,
                    PetscVec   u,
                    void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef TS  Ts = ref_TS(ts)
    cdef Vec Vu = ref_Vec(u)
    cdef object monitorlist = Ts.get_attr('__monitor__')
    if monitorlist is None: return 0
    for (monitor, args, kargs) in monitorlist:
        monitor(Ts, toInt(step), toReal(time), Vu, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef int TS_PreStep(PetscTS ts) except PETSC_ERR_PYTHON with gil:
    cdef TS Ts = ref_TS(ts)
    (prestep, args, kargs) = Ts.get_attr('__prestep__')
    prestep(Ts, *args, **kargs)
    return 0

cdef int TS_PostStep(PetscTS ts) except PETSC_ERR_PYTHON with gil:
    cdef TS Ts = ref_TS(ts)
    (poststep, args, kargs) = Ts.get_attr('__poststep__')
    poststep(Ts, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------
