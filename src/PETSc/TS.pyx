# -----------------------------------------------------------------------------

class TSType(object):
    # native
    EULER    = S_(TSEULER)
    BEULER   = S_(TSBEULER)
    PSEUDO   = S_(TSPSEUDO)
    CN       = S_(TSCN)
    SUNDIALS = S_(TSSUNDIALS)
    RK       = S_(TSRK)
    PYTHON   = S_(TSPYTHON)
    THETA    = S_(TSTHETA)
    GL       = S_(TSGL)
    SSP      = S_(TSSSP)
    #
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
        cdef const_char *cval = NULL
        ts_type = str2bytes(ts_type, &cval)
        CHKERR( TSSetType(self.ts, cval) )

    def getType(self):
        cdef PetscTSType cval = NULL
        CHKERR( TSGetType(self.ts, &cval) )
        return bytes2str(cval)

    def setProblemType(self, ptype):
        CHKERR( TSSetProblemType(self.ts, ptype) )

    def getProblemType(self):
        cdef PetscTSProblemType ptype
        CHKERR( TSGetProblemType(self.ts, &ptype) )
        return ptype

    def setOptionsPrefix(self, prefix):
        cdef const_char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( TSSetOptionsPrefix(self.ts, cval) )

    def getOptionsPrefix(self):
        cdef const_char *cval = NULL
        CHKERR( TSGetOptionsPrefix(self.ts, &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( TSSetFromOptions(self.ts) )

    # --- application context ---

    def setAppCtx(self, appctx):
        self.set_attr('__appctx__', appctx)

    def getAppCtx(self):
        return self.get_attr('__appctx__')

    # --- user LHS Matrix routines ---

    def setLHSMatrix(self, Mat Alhs not None,
                     lhsmatrix=None, args=None, kargs=None):
        cdef PetscMatStructure matstr = MAT_DIFFERENT_NONZERO_PATTERN # XXX
        if lhsmatrix is None:
            CHKERR( TSSetMatrices(self.ts, NULL, NULL,
                                  Alhs.mat, NULL, matstr, NULL) )
            self.set_attr('__lhsmatrix__', None)
        else:
            CHKERR( TSSetMatrices(self.ts, NULL, NULL,
                                  Alhs.mat, TS_LHSMatrix, matstr, NULL) )
            if args is None: args = ()
            if kargs is None: kargs = {}
            self.set_attr('__lhsmatrix__', (lhsmatrix, args, kargs))

    def setRHSMatrix(self, Mat Arhs not None,
                     rhsmatrix=None, args=None, kargs=None):
        cdef PetscMatStructure matstr = MAT_DIFFERENT_NONZERO_PATTERN # XXX
        if rhsmatrix is None:
            CHKERR( TSSetMatrices(self.ts, Arhs.mat, NULL,
                                  NULL, NULL, matstr, NULL) )
            self.set_attr('__rhsmatrix__', None)
        else:
            CHKERR( TSSetMatrices(self.ts, Arhs.mat, TS_RHSMatrix,
                                  NULL, NULL, matstr, NULL) )
            if args is None: args = ()
            if kargs is None: kargs = {}
            self.set_attr('__rhsmatrix__', (rhsmatrix, args, kargs))

    # --- user RHS Function/Jacobian routines ---

    def setRHSFunction(self, function, Vec f not None, args=None, kargs=None):
        cdef PetscVec fvec = NULL
        if f is not None: fvec = f.vec
        CHKERR( TSSetRHSFunction(self.ts, fvec, TS_RHSFunction, NULL) )
        if args is None: args = ()
        if kargs is None: kargs = {}
        self.set_attr('__rhsfunction__', (function, args, kargs))

    def setRHSJacobian(self, jacobian, Mat J, Mat P=None, args=None, kargs=None):
        cdef PetscMat Jmat = NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        CHKERR( TSSetRHSJacobian(self.ts, Jmat, Pmat, TS_RHSJacobian, NULL) )
        if args is None: args = ()
        if kargs is None: kargs = {}
        self.set_attr('__rhsjacobian__', (jacobian, args, kargs))

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
        cdef object function = self.get_attr('__rhsfunction__')
        return (f, function)

    def getRHSJacobian(self):
        cdef Mat J = Mat(), P = Mat()
        CHKERR( TSGetRHSJacobian(self.ts, &J.mat, &P.mat, NULL, NULL) )
        PetscIncref(<PetscObject>J.mat)
        PetscIncref(<PetscObject>P.mat)
        cdef object jacobian = self.get_attr('__rhsjacobian__')
        return (J, P, jacobian)

    # --- user Implicit Function/Jacobian routines ---

    def setIFunction(self, function, Vec f not None, args=None, kargs=None):
        cdef PetscVec fvec=NULL
        if f is not None: fvec = f.vec
        CHKERR( TSSetIFunction(self.ts, fvec, TS_IFunction, NULL) )
        if args is None: args = ()
        if kargs is None: kargs = {}
        self.set_attr('__ifunction__', (function, args, kargs))

    def setIJacobian(self, jacobian, Mat J, Mat P=None, args=None, kargs=None):
        cdef PetscMat Jmat = NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        CHKERR( TSSetIJacobian(self.ts, Jmat, Pmat, TS_IJacobian, NULL) )
        if args is None: args = ()
        if kargs is None: kargs = {}
        self.set_attr('__ijacobian__', (jacobian, args, kargs))
        if Pmat != NULL:
            CHKERR( PetscObjectCompose(
                    <PetscObject>self.ts,"__ijacpmat__", <PetscObject>Pmat) )

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
        cdef object function = self.get_attr('__ifunction__')
        return function

    def getIJacobian(self):
        cdef Mat J = Mat(), P = Mat()
        CHKERR( TSGetIJacobian(self.ts, &J.mat, &P.mat, NULL, NULL) )
        PetscIncref(<PetscObject>J.mat)
        PetscIncref(<PetscObject>P.mat)
        cdef object jacobian = self.get_attr('__ijacobian__')
        return (J, P, jacobian)

    # --- solution ---

    def setSolution(self, Vec u not None):
        CHKERR( TSSetSolution(self.ts, u.vec) )

    def getSolution(self):
        cdef Vec u = Vec()
        CHKERR( TSGetSolution(self.ts, &u.vec) )
        PetscIncref(<PetscObject>u.vec)
        return u

    # --- inner solver ---

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

    # --- finite diferences ---

    def setUseFD(self, flag=True):
        cdef PetscBool cflag = flag
        CHKERR( TSSetUseFDColoring(self.ts, cflag) )

    def getUseFD(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( TSGetUseFDColoring(self.ts, &flag) )
        return <bint> flag

    # --- customization ---

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

    # --- monitoring ---

    def setMonitor(self, monitor, args=None, kargs=None):
        cdef object monitorlist = None
        if monitor is not None:
            CHKERR( TSMonitorSet(self.ts, TS_Monitor, NULL, NULL) )
            monitorlist = self.get_attr('__monitor__')
            if monitorlist is None: monitorlist = []
            if args is None: args = ()
            if kargs is None: kargs = {}
            monitorlist.append((monitor, args, kargs))
        self.set_attr('__monitor__', monitorlist)

    def getMonitor(self):
        return self.get_attr('__monitor__')

    def callMonitor(self, step, time, Vec u=None):
        cdef PetscInt  ival = asInt(step)
        cdef PetscReal rval = asReal(time)
        cdef PetscVec  uvec = NULL
        if u is not None: uvec = u.vec
        if uvec == NULL:
            CHKERR( TSGetSolution(self.ts, &uvec) )
        CHKERR( TSMonitorCall(self.ts, ival, rval, uvec) )

    def cancelMonitor(self):
        CHKERR( TSMonitorCancel(self.ts) )
        self.set_attr('__monitor__', None)

    # --- solving ---

    def setPreStep(self, prestep, args=None, kargs=None):
        if prestep is not None:
            CHKERR( TSSetPreStep(self.ts, TS_PreStep) )
            if args is None: args = ()
            if kargs is None: kargs = {}
            self.set_attr('__prestep__', (prestep, args, kargs))
        else:
            CHKERR( TSSetPreStep(self.ts, NULL) )
            self.set_attr('__prestep__', None)

    def getPreStep(self, prestep):
        return self.get_attr('__prestep__')

    def setPostStep(self, poststep, args=None, kargs=None):
        if poststep is not None:
            CHKERR( TSSetPostStep(self.ts, TS_PostStep) )
            if args is None: args = ()
            if kargs is None: kargs = {}
            self.set_attr('__poststep__', (poststep, args, kargs))
        else:
            CHKERR( TSSetPostStep(self.ts, NULL) )
            self.set_attr('__poststep__', None)

    def getPostStep(self):
        return self.get_attr('__poststep__')

    def setUp(self):
        CHKERR( TSSetUp(self.ts) )

    def solve(self, Vec u not None):
        CHKERR( TSSolve(self.ts, u.vec) )

    # --- Python ---

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
        cdef const_char *cval = NULL
        py_type = str2bytes(py_type, &cval)
        CHKERR( TSPythonSetType(self.ts, cval) )

    # --- Theta ---

    def setTheta(self, theta):
        cdef PetscReal rval = asReal(theta)
        CHKERR( TSThetaSetTheta(self.ts, rval) )

    def getTheta(self):
        cdef PetscReal rval = 0
        CHKERR( TSThetaGetTheta(self.ts, &rval) )
        return toReal(rval)

    # --- application context ---

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
