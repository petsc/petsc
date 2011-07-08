# --------------------------------------------------------------------

class SNESType(object):
    LS      = S_(SNESLS)
    TR      = S_(SNESTR)
    PYTHON  = S_(SNESPYTHON)
    TEST    = S_(SNESTEST)
    PICARD  = S_(SNESPICARD)
    KSPONLY = S_(SNESKSPONLY)
    VI      = S_(SNESVI)

class SNESConvergedReason(object):
    # iterating
    CONVERGED_ITERATING      = SNES_CONVERGED_ITERATING
    ITERATING                = SNES_CONVERGED_ITERATING
    # converged
    CONVERGED_FNORM_ABS      = SNES_CONVERGED_FNORM_ABS
    CONVERGED_FNORM_RELATIVE = SNES_CONVERGED_FNORM_RELATIVE
    CONVERGED_PNORM_RELATIVE = SNES_CONVERGED_PNORM_RELATIVE
    CONVERGED_ITS            = SNES_CONVERGED_ITS
    CONVERGED_TR_DELTA       = SNES_CONVERGED_TR_DELTA
    # diverged
    DIVERGED_FUNCTION_DOMAIN = SNES_DIVERGED_FUNCTION_DOMAIN
    DIVERGED_FUNCTION_COUNT  = SNES_DIVERGED_FUNCTION_COUNT
    DIVERGED_FNORM_NAN       = SNES_DIVERGED_FNORM_NAN
    DIVERGED_MAX_IT          = SNES_DIVERGED_MAX_IT
    DIVERGED_LINE_SEARCH     = SNES_DIVERGED_LINE_SEARCH
    DIVERGED_LOCAL_MIN       = SNES_DIVERGED_LOCAL_MIN

# --------------------------------------------------------------------

cdef class SNES(Object):

    Type = SNESType
    ConvergedReason = SNESConvergedReason

    # --- xxx ---

    def __cinit__(self):
        self.obj  = <PetscObject*> &self.snes
        self.snes = NULL

    # --- xxx ---

    def view(self, Viewer viewer=None):
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( SNESView(self.snes, cviewer) )

    def destroy(self):
        CHKERR( SNESDestroy(&self.snes) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSNES newsnes = NULL
        CHKERR( SNESCreate(ccomm, &newsnes) )
        PetscCLEAR(self.obj); self.snes = newsnes
        return self

    def setType(self, snes_type):
        cdef PetscSNESType cval = NULL
        snes_type = str2bytes(snes_type, &cval)
        CHKERR( SNESSetType(self.snes, cval) )

    def getType(self):
        cdef PetscSNESType cval = NULL
        CHKERR( SNESGetType(self.snes, &cval) )
        return bytes2str(cval)

    def setOptionsPrefix(self, prefix):
        cdef const_char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( SNESSetOptionsPrefix(self.snes, cval) )

    def getOptionsPrefix(self):
        cdef const_char *cval = NULL
        CHKERR( SNESGetOptionsPrefix(self.snes, &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( SNESSetFromOptions(self.snes) )

    # --- application context ---

    def setAppCtx(self, appctx):
        self.set_attr('__appctx__', appctx)

    def getAppCtx(self):
        return self.get_attr('__appctx__')

    # --- discretization space ---

    def getDM(self):
        cdef PetscDM newdm = NULL
        CHKERR( SNESGetDM(self.snes, &newdm) )
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        PetscINCREF(dm.obj)
        return dm

    def setDM(self, DM dm not None):
        CHKERR( SNESSetDM(self.snes, dm.dm) )

    # --- user Function/Jacobian routines ---

    def setInitialGuess(self, initialguess, args=None, kargs=None):
        CHKERR( SNESSetInitialGuess(self.snes, SNES_InitialGuess, NULL) )
        if args is None: args = ()
        if kargs is None: kargs = {}
        self.set_attr('__initialguess__', (initialguess, args, kargs))

    def getInitialGuess(self):
        return self.get_attr('__initialguess__')

    def setFunction(self, function, Vec f, args=None, kargs=None):
        cdef PetscVec fvec=NULL
        if f is not None: fvec = f.vec
        CHKERR( SNESSetFunction(self.snes, fvec, SNES_Function, NULL) )
        if args is None: args = ()
        if kargs is None: kargs = {}
        self.set_attr('__function__', (function, args, kargs))

    def getFunction(self):
        cdef Vec f = Vec()
        CHKERR( SNESGetFunction(self.snes, &f.vec, NULL, NULL) )
        PetscINCREF(f.obj)
        cdef object function = self.get_attr('__function__')
        return (f, function)

    def setUpdate(self, update, args=None, kargs=None):
        if update is not None:
            CHKERR( SNESSetUpdate(self.snes, SNES_Update) )
            if args is None: args = ()
            if kargs is None: kargs = {}
            self.set_attr('__update__', (update, args, kargs))
        else:
            CHKERR( SNESSetUpdate(self.snes, NULL) )
            self.set_attr('__update__', None)

    def getUpdate(self):
        return self.get_attr('__update__')

    def setJacobian(self, jacobian, Mat J, Mat P=None, args=None, kargs=None):
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        CHKERR( SNESSetJacobian(self.snes, Jmat, Pmat, SNES_Jacobian, NULL) )
        if args is None: args = ()
        if kargs is None: kargs = {}
        self.set_attr('__jacobian__', (jacobian, args, kargs))

    def getJacobian(self):
        cdef Mat J = Mat()
        cdef Mat P = Mat()
        CHKERR( SNESGetJacobian(self.snes, &J.mat, &P.mat, NULL, NULL) )
        PetscINCREF(J.obj)
        PetscINCREF(P.obj)
        cdef object jacobian = self.get_attr('__jacobian__')
        return (J, P, jacobian)

    def computeFunction(self, Vec x not None, Vec f not None):
        CHKERR( SNESComputeFunction(self.snes, x.vec, f.vec) )

    def computeJacobian(self, Vec x not None, Mat J not None, Mat P=None):
        cdef PetscMat *jmat = &J.mat, *pmat = &J.mat
        if P is not None: pmat = &P.mat
        cdef PetscMatStructure flag = MAT_DIFFERENT_NONZERO_PATTERN
        CHKERR( SNESComputeJacobian(self.snes, x.vec, jmat, pmat, &flag) )
        return flag

    # --- tolerances and convergence ---

    def setTolerances(self, rtol=None, atol=None, stol=None, max_it=None):
        cdef PetscReal crtol, catol, cstol
        crtol = catol = cstol = PETSC_DEFAULT
        cdef PetscInt cmaxit = PETSC_DEFAULT
        if rtol   is not None: crtol  = asReal(rtol)
        if atol   is not None: catol  = asReal(atol)
        if stol   is not None: cstol  = asReal(stol)
        if max_it is not None: cmaxit = asInt(max_it)
        CHKERR( SNESSetTolerances(self.snes, catol, crtol, cstol,
                                  cmaxit, PETSC_DEFAULT) )

    def getTolerances(self):
        cdef PetscReal crtol=0, catol=0, cstol=0
        cdef PetscInt cmaxit=0
        CHKERR( SNESGetTolerances(self.snes, &catol, &crtol, &cstol,
                                  &cmaxit, NULL) )
        return (toReal(crtol), toReal(catol), toReal(cstol), toInt(cmaxit))

    def setConvergenceTest(self, converged, args=None, kargs=None):
        if converged is not None:
            CHKERR( SNESSetConvergenceTest(
                    self.snes, SNES_Converged, NULL, NULL) )
            if args is None: args = ()
            if kargs is None: kargs = {}
            self.set_attr('__converged__', (converged, args, kargs))
        else:
            CHKERR( SNESSetConvergenceTest(
                    self.snes, SNESDefaultConverged, NULL, NULL) )
            self.set_attr('__converged__', None)

    def getConvergenceTest(self):
        return self.get_attr('__converged__')

    def callConvergenceTest(self, its, xnorm, ynorm, fnorm):
        cdef PetscInt  ival  = asInt(its)
        cdef PetscReal rval1 = asReal(xnorm)
        cdef PetscReal rval2 = asReal(ynorm)
        cdef PetscReal rval3 = asReal(fnorm)
        cdef PetscSNESConvergedReason reason = SNES_CONVERGED_ITERATING
        CHKERR( SNESConvergenceTestCall(self.snes, ival,
                                        rval1, rval2, rval3, &reason) )
        return reason

    def setConvergenceHistory(self, length=None, reset=False):
        cdef PetscReal *rdata = NULL
        cdef PetscInt  *idata = NULL
        cdef PetscInt   size = 1000
        cdef PetscBool flag = PETSC_FALSE
        if   length is True:     pass
        elif length is not None: size = asInt(length)
        if size < 0: size = 1000
        if reset: flag = PETSC_TRUE
        cdef object rhist = oarray_r(empty_r(size), NULL, &rdata)
        cdef object ihist = oarray_i(empty_i(size), NULL, &idata)
        self.set_attr('__history__', (rhist, ihist))
        CHKERR( SNESSetConvergenceHistory(self.snes, rdata, idata, size, flag) )

    def getConvergenceHistory(self):
        cdef PetscReal *rdata = NULL
        cdef PetscInt  *idata = NULL
        cdef PetscInt   size = 0
        CHKERR( SNESGetConvergenceHistory(self.snes, &rdata, &idata, &size) )
        cdef object rhist = array_r(size, rdata)
        cdef object ihist = array_i(size, idata)
        return (rhist, ihist)

    def logConvergenceHistory(self, its, norm, linear_its=0):
        cdef PetscInt  ival1 = asInt(its)
        cdef PetscReal rval  = asReal(norm)
        cdef PetscInt  ival2 = asInt(linear_its)
        CHKERR( SNESLogConvergenceHistory(self.snes, ival1, rval, ival2) )

    # --- monitoring ---

    def setMonitor(self, monitor, args=None, kargs=None):
        cdef object monitorlist = None
        if monitor is not None:
            CHKERR( SNESMonitorSet(self.snes, SNES_Monitor, NULL, NULL) )
            monitorlist = self.get_attr('__monitor__')
            if monitorlist is None: monitorlist = []
            if args is None: args = ()
            if kargs is None: kargs = {}
            monitorlist.append((monitor, args, kargs))
        self.set_attr('__monitor__', monitorlist)

    def getMonitor(self):
        return self.get_attr('__monitor__')

    def cancelMonitor(self):
        CHKERR( SNESMonitorCancel(self.snes) )
        self.set_attr('__monitor__', None)

    def monitor(self, its, rnorm):
        cdef PetscInt  ival = asInt(its)
        cdef PetscReal rval = asReal(rnorm)
        CHKERR( SNESMonitor(self.snes, ival, rval) )

    # --- more tolerances ---

    def setMaxFunctionEvaluations(self, max_funcs):
        cdef PetscReal r = PETSC_DEFAULT
        cdef PetscInt  i = PETSC_DEFAULT
        cdef PetscInt ival = asInt(max_funcs)
        CHKERR( SNESSetTolerances(self.snes, r, r, r, i, ival) )

    def getMaxFunctionEvaluations(self):
        cdef PetscReal *r = NULL
        cdef PetscInt  *i = NULL
        cdef PetscInt ival = 0
        CHKERR( SNESGetTolerances(self.snes, r, r, r, i, &ival) )
        return toInt(ival)

    def getFunctionEvaluations(self):
        cdef PetscInt ival = 0
        CHKERR( SNESGetNumberFunctionEvals(self.snes, &ival) )
        return toInt(ival)

    def setMaxNonlinearStepFailures(self, max_fails):
        cdef PetscInt ival = asInt(max_fails)
        CHKERR( SNESSetMaxNonlinearStepFailures(self.snes, ival) )

    def getMaxNonlinearStepFailures(self):
        cdef PetscInt ival = 0
        CHKERR( SNESGetMaxNonlinearStepFailures(self.snes, &ival) )
        return toInt(ival)

    def getNonlinearStepFailures(self):
        cdef PetscInt ival = 0
        CHKERR( SNESGetNonlinearStepFailures(self.snes, &ival) )
        return toInt(ival)

    def setMaxLinearSolveFailures(self, max_fails):
        cdef PetscInt ival = asInt(max_fails)
        CHKERR( SNESSetMaxLinearSolveFailures(self.snes, ival) )

    def getMaxLinearSolveFailures(self):
        cdef PetscInt ival = 0
        CHKERR( SNESGetMaxLinearSolveFailures(self.snes, &ival) )
        return toInt(ival)

    def getLinearSolveFailures(self):
        cdef PetscInt ival = 0
        CHKERR( SNESGetLinearSolveFailures(self.snes, &ival) )
        return toInt(ival)

    # --- solving ---

    def setUp(self):
        CHKERR( SNESSetUp(self.snes) )

    def reset(self):
        CHKERR( SNESReset(self.snes) )

    def solve(self, Vec b, Vec x not None):
        cdef PetscVec rhs = NULL
        if b is not None: rhs = (<Vec>b).vec
        CHKERR( SNESSolve(self.snes, rhs, x.vec) )

    def setConvergedReason(self, reason):
        cdef PetscSNESConvergedReason eval = reason
        CHKERR( SNESSetConvergedReason(self.snes, eval) )

    def getConvergedReason(self):
        cdef PetscSNESConvergedReason reason = SNES_CONVERGED_ITERATING
        CHKERR( SNESGetConvergedReason(self.snes, &reason) )
        return reason

    def setIterationNumber(self, its):
        cdef PetscInt ival = asInt(its)
        CHKERR( SNESSetIterationNumber(self.snes, ival) )

    def getIterationNumber(self):
        cdef PetscInt ival = 0
        CHKERR( SNESGetIterationNumber(self.snes, &ival) )
        return toInt(ival)

    def setFunctionNorm(self, norm):
        cdef PetscReal rval = asReal(norm)
        CHKERR( SNESSetFunctionNorm(self.snes, rval) )

    def getFunctionNorm(self):
        cdef PetscReal rval = 0
        CHKERR( SNESGetFunctionNorm(self.snes, &rval) )
        return toReal(rval)

    def getLinearSolveIterations(self):
        cdef PetscInt ival = 0
        CHKERR( SNESGetLinearSolveIterations(self.snes, &ival) )
        return toInt(ival)

    def getRhs(self):
        cdef Vec vec = Vec()
        CHKERR( SNESGetRhs(self.snes, &vec.vec) )
        PetscINCREF(vec.obj)
        return vec

    def getSolution(self):
        cdef Vec vec = Vec()
        CHKERR( SNESGetSolution(self.snes, &vec.vec) )
        PetscINCREF(vec.obj)
        return vec

    def getSolutionUpdate(self):
        cdef Vec vec = Vec()
        CHKERR( SNESGetSolutionUpdate(self.snes, &vec.vec) )
        PetscINCREF(vec.obj)
        return vec

    # --- linear solver ---

    def setKSP(self, KSP ksp not None):
        CHKERR( SNESSetKSP(self.snes, ksp.ksp) )

    def getKSP(self):
        cdef KSP ksp = KSP()
        CHKERR( SNESGetKSP(self.snes, &ksp.ksp) )
        PetscINCREF(ksp.obj)
        return ksp

    def setUseEW(self, flag=True, *targs, **kargs):
        cdef PetscBool bval = flag
        CHKERR( SNESKSPSetUseEW(self.snes, bval) )
        if targs or kargs: self.setParamsEW(*targs, **kargs)

    def getUseEW(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( SNESKSPGetUseEW(self.snes, &flag) )
        return <bint> flag

    def setParamsEW(self, version=None,
                    rtol_0=None, rtol_max=None,
                    gamma=None, alpha=None, alpha2=None,
                    threshold=None):
        cdef PetscInt  cversion   = PETSC_DEFAULT
        cdef PetscReal crtol_0    = PETSC_DEFAULT
        cdef PetscReal crtol_max  = PETSC_DEFAULT
        cdef PetscReal cgamma     = PETSC_DEFAULT
        cdef PetscReal calpha     = PETSC_DEFAULT
        cdef PetscReal calpha2    = PETSC_DEFAULT
        cdef PetscReal cthreshold = PETSC_DEFAULT
        if version   is not None: cversion   = asInt(version)
        if rtol_0    is not None: crtol_0    = asReal(rtol_0)
        if rtol_max  is not None: crtol_max  = asReal(rtol_max)
        if gamma     is not None: cgamma     = asReal(gamma)
        if alpha     is not None: calpha     = asReal(alpha)
        if alpha2    is not None: calpha2    = asReal(alpha2)
        if threshold is not None: cthreshold = asReal(threshold)
        CHKERR( SNESKSPSetParametersEW(
            self.snes, cversion, crtol_0, crtol_max,
            cgamma, calpha, calpha2, cthreshold) )

    def getParamsEW(self):
        cdef PetscInt  version=0
        cdef PetscReal rtol_0=0, rtol_max=0
        cdef PetscReal gamma=0, alpha=0, alpha2=0
        cdef PetscReal threshold=0
        CHKERR( SNESKSPGetParametersEW(
            self.snes, &version, &rtol_0, &rtol_max,
            &gamma, &alpha, &alpha2, &threshold) )
        return {'version'   : toInt(version),
                'rtol_0'    : toReal(rtol_0),
                'rtol_max'  : toReal(rtol_max),
                'gamma'     : toReal(gamma),
                'alpha'     : toReal(alpha),
                'alpha2'    : toReal(alpha2),
                'threshold' : toReal(threshold),}

    # --- matrix free / finite diferences ---

    def setUseMF(self, flag=True):
        cdef PetscBool bval = flag
        CHKERR( SNESSetUseMFFD(self.snes, bval) )

    def getUseMF(self):
        cdef PetscBool bval = PETSC_FALSE
        CHKERR( SNESGetUseMFFD(self.snes, &bval) )
        return <bint> bval

    def setUseFD(self, flag=True):
        cdef PetscBool bval = flag
        CHKERR( SNESSetUseFDColoring(self.snes, bval) )

    def getUseFD(self):
        cdef PetscBool bval = PETSC_FALSE
        CHKERR( SNESGetUseFDColoring(self.snes, &bval) )
        return <bint> bval

    # --- Python ---

    def createPython(self, context=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSNES newsnes = NULL
        CHKERR( SNESCreate(ccomm, &newsnes) )
        PetscCLEAR(self.obj); self.snes = newsnes
        CHKERR( SNESSetType(self.snes, SNESPYTHON) )
        CHKERR( SNESPythonSetContext(self.snes, <void*>context) )
        return self

    def setPythonContext(self, context):
        CHKERR( SNESPythonSetContext(self.snes, <void*>context) )

    def getPythonContext(self):
        cdef void *context = NULL
        CHKERR( SNESPythonGetContext(self.snes, &context) )
        if context == NULL: return None
        else: return <object> context

    def setPythonType(self, py_type):
        cdef const_char *cval = NULL
        py_type = str2bytes(py_type, &cval)
        CHKERR( SNESPythonSetType(self.snes, cval) )

    # --- application context ---

    property appctx:
        def __get__(self):
            return self.getAppCtx()
        def __set__(self, value):
            self.setAppCtx(value)

    # --- discretization space ---

    property dm:
        def __get__(self):
            return self.getDM()
        def __set__(self, value):
            self.setDM(value)

    # --- vectors ---

    property vec_sol:
        def __get__(self):
            return self.getSolution()

    property vec_upd:
        def __get__(self):
            return self.getSolutionUpdate()

    property vec_rhs:
        def __get__(self):
            return self.getRhs()

    # --- linear solver ---

    property ksp:
        def __get__(self):
            return self.getKSP()
        def __set__(self, value):
            self.setKSP(value)

    property use_ew:
        def __get__(self):
            return self.getUseEW()
        def __set__(self, value):
            self.setUseEW(value)

    # --- tolerances ---

    property rtol:
        def __get__(self):
            return self.getTolerances()[0]
        def __set__(self, value):
            self.setTolerances(rtol=value)

    property atol:
        def __get__(self):
            return self.getTolerances()[1]
        def __set__(self, value):
            self.setTolerances(atol=value)

    property stol:
        def __get__(self):
            return self.getTolerances()[2]
        def __set__(self, value):
            self.setTolerances(stol=value)

    property max_it:
        def __get__(self):
            return self.getTolerances()[3]
        def __set__(self, value):
            self.setTolerances(max_it=value)

    # --- more tolerances ---

    property max_funcs:
        def __get__(self):
            return self.getMaxFunctionEvaluations()
        def __set__(self, value):
            self.setMaxFunctionEvaluations(value)

    # --- iteration ---

    property its:
        def __get__(self):
            return self.getIterationNumber()
        def __set__(self, value):
            self.setIterationNumber(value)

    property norm:
        def __get__(self):
            return self.getFunctionNorm()
        def __set__(self, value):
            self.setFunctionNorm(value)

    property history:
        def __get__(self):
            return self.getConvergenceHistory()

    # --- convergence ---

    property reason:
        def __get__(self):
            return self.getConvergedReason()
        def __set__(self, value):
            self.setConvergedReason(value)

    property iterating:
        def __get__(self):
            return self.reason == 0

    property converged:
        def __get__(self):
            return self.reason > 0

    property diverged:
        def __get__(self):
            return self.reason < 0

    # --- matrix free / finite diferences ---

    property use_mf:
        def __get__(self):
            return self.getUseMF()
        def __set__(self, value):
            self.setUseMF(value)

    property use_fd:
        def __get__(self):
            return self.getUseFD()
        def __set__(self, value):
            self.setUseFD(value)

# --------------------------------------------------------------------

del SNESType
del SNESConvergedReason

# --------------------------------------------------------------------
