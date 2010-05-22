# -----------------------------------------------------------------------------

class TSType(object):
    # native
    EULER           = TSEULER
    BEULER          = TSBEULER
    CN              = TSCN
    RK              = TSRK
    PSEUDO          = TSPSEUDO
    SUNDIALS        = TSSUNDIALS
    THETA           = TSTHETA
    GL              = TSGL
    SSP             = TSSSP
    #
    PYTHON = TSPYTHON
    # aliases
    FE = EULER
    BE = BEULER
    CRANK_NICOLSON = CN
    RUNGE_KUTTA    = RK

class TSProblemType(object):
    LINEAR    = TS_LINEAR
    NONLINEAR = TS_NONLINEAR

# -----------------------------------------------------------------------------

cdef class TS(Object):

    Type = TSType
    ProblemType = TSProblemType

    # --- xxx ---

    def __cinit__(self):
        self.obj = <PetscObject*> &self.ts
        self.ts = NULL

    # --- xxx ---

    def view(self, Viewer viewer=None):
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( TSView(self.ts, cviewer) )

    def destroy(self):
        CHKERR( TSDestroy(self.ts) )
        self.ts = NULL
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscTS newts = NULL
        CHKERR( TSCreate(ccomm, &newts) )
        PetscCLEAR(self.obj); self.ts = newts
        return self

    def setType(self, ts_type):
        CHKERR( TSSetType(self.ts, str2cp(ts_type)) )

    def getType(self):
        cdef PetscTSType ts_type = NULL
        CHKERR( TSGetType(self.ts, &ts_type) )
        return cp2str(ts_type)

    def setProblemType(self, ptype):
        CHKERR( TSSetProblemType(self.ts, ptype) )

    def getProblemType(self):
        cdef PetscTSProblemType ptype
        CHKERR( TSGetProblemType(self.ts, &ptype) )
        return ptype

    def setOptionsPrefix(self, prefix):
        CHKERR( TSSetOptionsPrefix(self.ts, str2cp(prefix)) )

    def getOptionsPrefix(self):
        cdef const_char_p prefix = NULL
        CHKERR( TSGetOptionsPrefix(self.ts, &prefix) )
        return cp2str(prefix)

    def setFromOptions(self):
        CHKERR( TSSetFromOptions(self.ts) )

    # --- xxx ---

    def setAppCtx(self, appctx):
        Object_setAttr(<PetscObject>self.ts, '__appctx__', appctx)

    def getAppCtx(self):
        return Object_getAttr(<PetscObject>self.ts, '__appctx__')

    # --- xxx ---

    def setRHSFunction(self, function, Vec f not None, *args, **kargs):
        cdef PetscVec fvec = NULL
        if f is not None: fvec = f.vec
        TS_setRHSFunction(self.ts, fvec, (function, args, kargs))

    def setRHSJacobian(self, jacobian, Mat J, Mat P=None, *args, **kargs):
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        TS_setRHSJacobian(self.ts, Jmat, Pmat, (jacobian, args, kargs))

    def computeRHSFunction(self, t, Vec x not None, Vec f not None):
        cdef PetscReal time = asReal(t)
        CHKERR( TSComputeRHSFunction(self.ts, time, x.vec, f.vec) )

    def computeRHSJacobian(self, t, Vec x not None, Mat J not None, Mat P=None):
        cdef PetscReal time = asReal(t)
        cdef PetscMat *jmat = &J.mat, *pmat = &J.mat
        if P is not None: pmat = &P.mat
        cdef PetscMatStructure flag = MAT_DIFFERENT_NONZERO_PATTERN
        CHKERR( TSComputeRHSJacobian(self.ts, time, x.vec,
                                     jmat, pmat, &flag) )
        return flag

    def getRHSFunction(self):
        cdef Vec f = Vec()
        CHKERR( TSGetRHSFunction(self.ts, &f.vec, NULL, NULL) )
        PetscIncref(<PetscObject>f.vec)
        cdef object function = TS_getRHSFunction(self.ts)
        return (f, function)

    def getRHSJacobian(self):
        cdef Mat J = Mat(), P = Mat()
        CHKERR( TSGetRHSJacobian(self.ts, &J.mat, &P.mat, NULL, NULL) )
        PetscIncref(<PetscObject>J.mat)
        PetscIncref(<PetscObject>P.mat)
        cdef object jacobian = TS_getRHSJacobian(self.ts)
        return (J, P, jacobian)

    #

    def setIFunction(self, function, Vec f not None, *args, **kargs):
        cdef PetscVec fvec=NULL
        if f is not None: fvec = f.vec
        TS_setIFunction(self.ts, f.vec, (function, args, kargs))

    def setIJacobian(self, jacobian, Mat J, Mat P=None, *args, **kargs):
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        TS_setIJacobian(self.ts, Jmat, Pmat, (jacobian, args, kargs))

    def computeIFunction(self,
                         t, Vec x not None, Vec xdot not None,
                         Vec f not None):
        cdef PetscReal time = asReal(t)
        CHKERR( TSComputeIFunction(self.ts, time, x.vec, xdot.vec, f.vec) )

    def computeIJacobian(self,
                         t, Vec x not None, Vec xdot not None, a,
                         Mat J not None, Mat P=None):
        cdef PetscReal time  = asReal(t)
        cdef PetscReal shift = asReal(a)
        cdef PetscMat *jmat = &J.mat, *pmat = &J.mat
        if P is not None: pmat = &P.mat
        cdef PetscMatStructure flag = MAT_DIFFERENT_NONZERO_PATTERN
        CHKERR( TSComputeIJacobian(self.ts, time, x.vec, xdot.vec, shift,
                                   jmat, pmat, &flag) )
        return flag

    def getIFunction(self):
        cdef object function = TS_getIFunction(self.ts)
        return function

    def getIJacobian(self):
        cdef Mat J = Mat(), P = Mat()
        CHKERR( TSGetIJacobian(self.ts, &J.mat, &P.mat, NULL, NULL) )
        PetscIncref(<PetscObject>J.mat)
        PetscIncref(<PetscObject>P.mat)
        cdef object jacobian = TS_getIJacobian(self.ts)
        return (J, P, jacobian)

    #

    def setSolution(self, Vec u not None):
        CHKERR( TSSetSolution(self.ts, u.vec) )

    def getSolution(self):
        cdef Vec u = Vec()
        CHKERR( TSGetSolution(self.ts, &u.vec) )
        PetscIncref(<PetscObject>u.vec)
        return u
    #

    def getSNES(self):
        cdef SNES snes = SNES()
        CHKERR( TSGetSNES(self.ts, &snes.snes) )
        PetscIncref(<PetscObject>snes.snes)
        return snes

    def getKSP(self):
        cdef KSP ksp = KSP()
        CHKERR( TSGetKSP(self.ts, &ksp.ksp) )
        PetscIncref(<PetscObject>ksp.ksp)
        return ksp

    #

    def setUseFD(self, flag=True):
        cdef PetscTruth cflag = flag
        CHKERR( TSSetUseFDColoring(self.ts, cflag) )

    def getUseFD(self):
        cdef PetscTruth flag = PETSC_FALSE
        CHKERR( TSGetUseFDColoring(self.ts, &flag) )
        return <bint> flag

    # --- xxx ---

    def setTime(self, t):
        cdef PetscReal rval = asReal(t)
        CHKERR( TSSetTime(self.ts, rval) )

    def getTime(self):
        cdef PetscReal rval = 0
        CHKERR( TSGetTime(self.ts, &rval) )
        return toReal(rval)

    def setInitialTimeStep(self, initial_time, initial_time_step):
        cdef PetscReal rval1 = asReal(initial_time)
        cdef PetscReal rval2 = asReal(initial_time_step)
        CHKERR( TSSetInitialTimeStep(self.ts, rval1, rval2) )

    def setTimeStep(self, time_step):
        cdef PetscReal rval = asReal(time_step)
        CHKERR( TSSetTimeStep(self.ts, rval) )

    def getTimeStep(self):
        cdef PetscReal tstep = 0
        CHKERR( TSGetTimeStep(self.ts, &tstep) )
        return toReal(tstep)

    def setStepNumber(self, step_number):
        cdef PetscInt ival = asInt(step_number)
        CHKERR( TSSetTimeStepNumber(self.ts, ival) )

    def getStepNumber(self):
        cdef PetscInt ival = 0
        CHKERR( TSGetTimeStepNumber(self.ts, &ival) )
        return toInt(ival)

    def setMaxTime(self, max_time):
        cdef PetscInt  ival = 0
        cdef PetscReal rval = asReal(max_time)
        CHKERR( TSGetDuration(self.ts, &ival, NULL) )
        CHKERR( TSSetDuration(self.ts, ival, rval) )

    def getMaxTime(self):
        cdef PetscReal rval = 0
        CHKERR( TSGetDuration(self.ts, NULL, &rval) )
        return toReal(rval)

    def setMaxSteps(self, max_steps):
        cdef PetscInt  ival = asInt(max_steps)
        cdef PetscReal rval = 0
        CHKERR( TSGetDuration(self.ts, NULL, &rval) )
        CHKERR( TSSetDuration(self.ts, ival, rval) )

    def getMaxSteps(self):
        cdef PetscInt ival = 0
        CHKERR( TSGetDuration(self.ts, &ival, NULL) )
        return toInt(ival)

    def setDuration(self, max_time, max_steps=None):
        cdef PetscInt  ival = 0
        cdef PetscReal rval = 0
        CHKERR( TSGetDuration(self.ts, &ival, &rval) )
        if max_steps is not None: ival = asInt(max_steps)
        if max_time  is not None: rval = asReal(max_time)
        CHKERR( TSSetDuration(self.ts, ival, rval) )

    def getDuration(self):
        cdef PetscInt  ival = 0
        cdef PetscReal rval = 0
        CHKERR( TSGetDuration(self.ts, &ival, &rval) )
        return (toReal(rval), toInt(ival))

    #

    def setMonitor(self, monitor, *args, **kargs):
        if monitor is None: TS_setMonitor(self.ts, None)
        else: TS_setMonitor(self.ts, (monitor, args, kargs))

    def getMonitor(self):
        return TS_getMonitor(self.ts)

    def callMonitor(self, step, time, Vec u=None):
        cdef PetscInt  ival = asInt(step)
        cdef PetscReal rval = asReal(time)
        cdef PetscVec  uvec = NULL
        if u is not None: uvec = u.vec
        if uvec == NULL:
            ## CHKERR( TSGetSolutionUpdate(self.ts, &uvec) )
            if uvec == NULL:
                CHKERR( TSGetSolution(self.ts, &uvec) )
        CHKERR( TSMonitorCall(self.ts, ival, rval, uvec) )

    def cancelMonitor(self):
        CHKERR( TSMonitorCancel(self.ts) )
        TS_delMonitor(self.ts)

    #

    def setPreStep(self, prestep, *args, **kargs):
        if prestep is not None: prestep = (prestep, args, kargs)
        TS_setPreStep(self.ts, prestep)

    def getPreStep(self, prestep):
        return TS_getPreStep(self.ts)

    def setPostStep(self, poststep, *args, **kargs):
        if poststep is not None: prestep = (poststep, args, kargs)
        TS_setPostStep(self.ts, (poststep, args, kargs))

    def getPostStep(self):
        return TS_getPostStep(self.ts)

    def setUp(self):
        CHKERR( TSSetUp(self.ts) )

    def solve(self, Vec u not None):
        CHKERR( TSSolve(self.ts, u.vec) )

    # Python
    # ------

    def createPython(self, context=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscTS newts = NULL
        CHKERR( TSCreate(ccomm, &newts) )
        PetscCLEAR(self.obj); self.ts = newts
        CHKERR( TSSetType(self.ts, TSPYTHON) )
        CHKERR( TSPythonSetContext(self.ts, <void*>context) )
        return self

    def setPythonContext(self, context):
        CHKERR( TSPythonSetContext(self.ts, <void*>context) )

    def getPythonContext(self):
        cdef void *context = NULL
        CHKERR( TSPythonGetContext(self.ts, &context) )
        if context == NULL: return None
        else: return <object> context

    def setPythonType(self, py_type):
        CHKERR( TSPythonSetType(self.ts, str2cp(py_type)) )

    # Theta
    # -----

    def setTheta(self, theta):
        cdef PetscReal rval = asReal(theta)
        CHKERR( TSThetaSetTheta(self.ts, rval) )

    def getTheta(self):
        cdef PetscReal rval = 0
        CHKERR( TSThetaGetTheta(self.ts, &rval) )
        return toReal(rval)

    # --- xxx ---

    property appctx:
        def __get__(self):
            return self.getAppCtx()
        def __set__(self, value):
            self.setAppCtx(value)

    # --- xxx ---

    property problem_type:
        def __get__(self):
            return self.getProblemType()
        def __set__(self, value):
            self.setProblemType(value)

    property snes:
        def __get__(self):
            return self.getSNES()

    property ksp:
        def __get__(self):
            return self.getKSP()

    property vec_sol:
        def __get__(self):
            return self.getSolution()

    # --- xxx ---

    property time:
        def __get__(self):
            return self.getTime()
        def __set__(self, value):
            self.setTime(value)

    property time_step:
        def __get__(self):
            return self.getTimeStep()
        def __set__(self, value):
            self.setTimeStep(value)

    property step_number:
        def __get__(self):
            return self.getStepNumber()
        def __set__(self, value):
            self.setStepNumber(value)

    property max_time:
        def __get__(self):
            return self.getMaxTime()
        def __set__(self, value):
            self.setMaxTime(value)

    property max_steps:
        def __get__(self):
            return self.getMaxSteps()
        def __set__(self, value):
            self.setMaxSteps(value)

    # --- finite diferences ---

    property use_fd:
        def __get__(self):
            return self.getUseFD()
        def __set__(self, value):
            self.setUseFD(value)

# -----------------------------------------------------------------------------

del TSType
del TSProblemType

# -----------------------------------------------------------------------------
