# --------------------------------------------------------------------

class SNESType(object):
    LS = SNESLS
    TR = SNESTR
    #
    PYTHON = SNESPYTHON

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
    DIVERGED_LS_FAILURE      = SNES_DIVERGED_LS_FAILURE
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
        CHKERR( SNESDestroy(self.snes) )
        self.snes = NULL
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSNES newsnes = NULL
        CHKERR( SNESCreate(ccomm, &newsnes) )
        PetscCLEAR(self.obj); self.snes = newsnes
        return self

    def setType(self, snes_type):
        CHKERR( SNESSetType(self.snes, str2cp(snes_type)) )

    def getType(self):
        cdef PetscSNESType snes_type = NULL
        CHKERR( SNESGetType(self.snes, &snes_type) )
        return cp2str(snes_type)

    def setOptionsPrefix(self, prefix):
        CHKERR( SNESSetOptionsPrefix(self.snes, str2cp(prefix)) )

    def getOptionsPrefix(self):
        cdef const_char_p prefix = NULL
        CHKERR( SNESGetOptionsPrefix(self.snes, &prefix) )
        return cp2str(prefix)

    def setFromOptions(self):
        CHKERR( SNESSetFromOptions(self.snes) )

    # --- xxx ---

    def setAppCtx(self, appctx):
        Object_setAttr(<PetscObject>self.snes, "__appctx__", appctx)

    def getAppCtx(self):
        return Object_getAttr(<PetscObject>self.snes, "__appctx__")

    # --- xxx ---

    def setFunction(self, function, Vec f not None, *args, **kargs):
        SNES_setFun(self.snes, f.vec, (function, args, kargs))

    def getFunction(self):
        cdef Vec f = Vec()
        CHKERR( SNESGetFunction(self.snes, &f.vec, NULL, NULL) )
        PetscIncref(<PetscObject>f.vec)
        cdef object fun = SNES_getFun(self.snes)
        return (f, fun)

    def setUpdate(self, update, *args, **kargs):
        if update is None: SNES_setUpd(self.snes, None)
        else: SNES_setUpd(self.snes, (update, args, kargs))

    def getUpdate(self):
        return SNES_getUpd(self.snes)

    def setJacobian(self, jacobian, Mat J, Mat P=None, *args, **kargs):
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        SNES_setJac(self.snes, Jmat, Pmat, (jacobian, args, kargs))

    def getJacobian(self):
        cdef Mat J = Mat()
        cdef Mat P = Mat()
        CHKERR( SNESGetJacobian(self.snes, &J.mat, &P.mat, NULL, NULL) )
        PetscIncref(<PetscObject>J.mat)
        PetscIncref(<PetscObject>P.mat)
        cdef object jac = SNES_getJac(self.snes)
        return (J, P, jac)

    def computeFunction(self, Vec x not None, Vec f not None):
        CHKERR( SNESComputeFunction(self.snes, x.vec, f.vec) )

    def computeJacobian(self, Vec x not None, Mat J not None, Mat P=None):
        cdef PetscMat *jmat = &J.mat, *pmat = &J.mat
        if P is not None: pmat = &P.mat
        cdef PetscMatStructure flag = MAT_DIFFERENT_NONZERO_PATTERN
        CHKERR( SNESComputeJacobian(self.snes, x.vec, jmat, pmat, &flag) )
        return flag

    def setKSP(self, KSP ksp not None):
        CHKERR( SNESSetKSP(self.snes, ksp.ksp) )

    def getKSP(self):
        cdef KSP ksp = KSP()
        CHKERR( SNESGetKSP(self.snes, &ksp.ksp) )
        PetscIncref(<PetscObject>ksp.ksp)
        return ksp

    # --- xxx ---

    def setTolerances(self, rtol=None, atol=None, stol=None, max_it=None):
        cdef PetscReal crtol, catol, cstol
        crtol = catol = cstol = PETSC_DEFAULT
        cdef PetscInt cmaxit = PETSC_DEFAULT
        if rtol   is not None: crtol = asReal(rtol)
        if atol   is not None: catol = asReal(atol)
        if stol   is not None: cstol = asReal(stol)
        if max_it is not None: cmaxit = max_it
        CHKERR( SNESSetTolerances(self.snes, catol, crtol, cstol,
                                  cmaxit, PETSC_DEFAULT) )

    def getTolerances(self):
        cdef PetscReal crtol=0, catol=0, cstol=0
        cdef PetscInt cmaxit=0
        CHKERR( SNESGetTolerances(self.snes, &catol, &crtol, &cstol,
                                  &cmaxit, NULL) )
        return (toReal(crtol), toReal(catol), toReal(cstol), cmaxit)

    def setConvergenceTest(self, converged, *args, **kargs):
        if converged is None: SNES_setCnv(self.snes, None)
        else: SNES_setCnv(self.snes, (converged, args, kargs))

    def getConvergenceTest(self):
        return SNES_getCnv(self.snes)

    def callConvergenceTest(self, its, xnorm, ynorm, fnorm):
        cdef PetscInt  ival  = its
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
        cdef PetscTruth flag = PETSC_FALSE
        if length is not None: size = length
        if size < 0: size = 1000
        if reset: flag = PETSC_TRUE
        cdef ndarray rhist = oarray_r(empty_r(size), NULL, &rdata)
        cdef ndarray ihist = oarray_i(empty_i(size), NULL, &idata)
        Object_setAttr(<PetscObject>self.snes, "__history__", (rhist, ihist))
        CHKERR( SNESSetConvergenceHistory(self.snes, rdata, idata, size, flag) )

    def getConvergenceHistory(self):
        cdef PetscReal *rdata = NULL
        cdef PetscInt  *idata = NULL
        cdef PetscInt   size = 0
        CHKERR( SNESGetConvergenceHistory(self.snes, &rdata, &idata, &size) )
        cdef ndarray rhist = array_r(size, rdata)
        cdef ndarray ihist = array_i(size, idata)
        return (rhist, ihist)

    def logConvergenceHistory(self, its, norm, linear_its=0):
        cdef PetscInt  ival1 = its
        cdef PetscReal rval  = asReal(norm)
        cdef PetscInt  ival2 = linear_its
        CHKERR( SNESLogConvergenceHistory(self.snes, ival1, rval, ival2) )

    def setMonitor(self, monitor, *args, **kargs):
        if monitor is None: SNES_setMon(self.snes, None)
        else: SNES_setMon(self.snes, (monitor, args, kargs))

    def getMonitor(self):
        return SNES_getMon(self.snes)

    def callMonitor(self, its, rnorm):
        cdef PetscInt  ival = its
        cdef PetscReal rval = asReal(rnorm)
        CHKERR( SNESMonitorCall(self.snes, ival, rval) )

    def cancelMonitor(self):
        SNES_clsMon(self.snes)

    # --- xxx ---

    def setMaxFunctionEvaluations(self, max_funcs):
        cdef PetscReal r = PETSC_DEFAULT
        cdef PetscInt  i = PETSC_DEFAULT
        CHKERR( SNESSetTolerances(self.snes, r, r, r, i, max_funcs) )

    def getMaxFunctionEvaluations(self):
        cdef PetscReal *r = NULL
        cdef PetscInt  *i = NULL
        cdef PetscInt max_funcs = 0
        CHKERR( SNESGetTolerances(self.snes, r, r, r, i, &max_funcs) )

    def getFunctionEvaluations(self):
        cdef PetscInt nfuncs = 0
        CHKERR( SNESGetNumberFunctionEvals(self.snes, &nfuncs) )
        return nfuncs

    def setMaxNonlinearStepFailures(self, max_fails):
        CHKERR( SNESSetMaxNonlinearStepFailures(self.snes, max_fails) )

    def getMaxNonlinearStepFailures(self):
        cdef PetscInt max_fails = 0
        CHKERR( SNESGetMaxNonlinearStepFailures(self.snes, &max_fails) )
        return max_fails

    def getNonlinearStepFailures(self):
        cdef PetscInt fails = 0
        CHKERR( SNESGetNonlinearStepFailures(self.snes, &fails) )
        return fails

    def setMaxLinearSolveFailures(self, max_fails):
        CHKERR( SNESSetMaxLinearSolveFailures(self.snes, max_fails) )

    def getMaxLinearSolveFailures(self):
        cdef PetscInt max_fails = 0
        CHKERR( SNESGetMaxLinearSolveFailures(self.snes, &max_fails) )
        return max_fails

    def getLinearSolveFailures(self):
        cdef PetscInt fails = 0
        CHKERR( SNESGetLinearSolveFailures(self.snes, &fails) )
        return fails

    # --- xxx ---

    def setUp(self):
        CHKERR( SNESSetUp(self.snes) )

    def solve(self, b, Vec x not None):
        cdef PetscVec rhs = NULL
        if b is not None: rhs = (<Vec?>b).vec
        CHKERR( SNESSolve(self.snes, rhs, x.vec) )

    def setConvergedReason(self, reason):
        cdef PetscSNESConvergedReason eval = reason
        CHKERR( SNESSetConvergedReason(self.snes, eval) )

    def getConvergedReason(self):
        cdef PetscSNESConvergedReason reason = SNES_CONVERGED_ITERATING
        CHKERR( SNESGetConvergedReason(self.snes, &reason) )
        return reason

    def setIterationNumber(self, its):
        cdef PetscInt ival = its
        CHKERR( SNESSetIterationNumber(self.snes, ival) )

    def getIterationNumber(self):
        cdef PetscInt ival = 0
        CHKERR( SNESGetIterationNumber(self.snes, &ival) )
        return ival

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
        return ival

    def getRhs(self):
        cdef Vec vec = Vec()
        CHKERR( SNESGetRhs(self.snes, &vec.vec) )
        PetscIncref(<PetscObject>vec.vec)
        return vec

    def getSolution(self):
        cdef Vec vec = Vec()
        CHKERR( SNESGetSolution(self.snes, &vec.vec) )
        PetscIncref(<PetscObject>vec.vec)
        return vec

    def getSolutionUpdate(self):
        cdef Vec vec = Vec()
        CHKERR( SNESGetSolutionUpdate(self.snes, &vec.vec) )
        PetscIncref(<PetscObject>vec.vec)
        return vec

    # --- xxx ---

    def setUseEW(self,  flag, *targs, **kargs):
        CHKERR( SNESKSPSetUseEW(self.snes, flag) )
        if targs or kargs: self.setParamsEW(*targs, **kargs)

    def getUseEW(self):
        cdef PetscTruth flag = PETSC_FALSE
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
        if version   is not None: cversion   = version
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
        return {'version'   : toReal(version),
                'rtol_0'    : toReal(rtol_0),
                'rtol_max'  : toReal(rtol_max),
                'gamma'     : toReal(gamma),
                'alpha'     : toReal(alpha),
                'alpha2'    : toReal(alpha2),
                'threshold' : toReal(threshold),}

    def setUseMF(self, flag):
        CHKERR( SNESSetUseMFFD(self.snes, flag) )

    def getUseMF(self):
        cdef PetscTruth flag = PETSC_FALSE
        CHKERR( SNESGetUseMFFD(self.snes, &flag) )
        return <bint> flag

    def setUseFD(self, flag):
        CHKERR( SNESSetUseFDColoring(self.snes, flag) )

    def getUseFD(self):
        cdef PetscTruth flag = PETSC_FALSE
        CHKERR( SNESGetUseFDColoring(self.snes, &flag) )
        return <bint> flag

    # --- xxx ---

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

    # --- xxx ---

    property appctx:
        def __get__(self):
            return self.getAppCtx()
        def __set__(self, value):
            self.setAppCtx(value)

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
