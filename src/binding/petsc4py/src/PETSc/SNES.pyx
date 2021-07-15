# --------------------------------------------------------------------

class SNESType(object):
    NEWTONLS         = S_(SNESNEWTONLS)
    NEWTONTR         = S_(SNESNEWTONTR)
    PYTHON           = S_(SNESPYTHON)
    NRICHARDSON      = S_(SNESNRICHARDSON)
    KSPONLY          = S_(SNESKSPONLY)
    KSPTRANSPOSEONLY = S_(SNESKSPTRANSPOSEONLY)
    VINEWTONRSLS     = S_(SNESVINEWTONRSLS)
    VINEWTONSSLS     = S_(SNESVINEWTONSSLS)
    NGMRES           = S_(SNESNGMRES)
    QN               = S_(SNESQN)
    SHELL            = S_(SNESSHELL)
    NGS              = S_(SNESNGS)
    NCG              = S_(SNESNCG)
    FAS              = S_(SNESFAS)
    MS               = S_(SNESMS)
    NASM             = S_(SNESNASM)
    ANDERSON         = S_(SNESANDERSON)
    ASPIN            = S_(SNESASPIN)
    COMPOSITE        = S_(SNESCOMPOSITE)
    PATCH            = S_(SNESPATCH)

class SNESNormSchedule(object):
    # native
    NORM_DEFAULT            = SNES_NORM_DEFAULT
    NORM_NONE               = SNES_NORM_NONE
    NORM_ALWAYS             = SNES_NORM_ALWAYS
    NORM_INITIAL_ONLY       = SNES_NORM_INITIAL_ONLY
    NORM_FINAL_ONLY         = SNES_NORM_FINAL_ONLY
    NORM_INITIAL_FINAL_ONLY = SNES_NORM_INITIAL_FINAL_ONLY
    # aliases
    DEFAULT            = NORM_DEFAULT
    NONE               = NORM_NONE
    ALWAYS             = NORM_ALWAYS
    INITIAL_ONLY       = NORM_INITIAL_ONLY
    FINAL_ONLY         = NORM_FINAL_ONLY
    INITIAL_FINAL_ONLY = NORM_INITIAL_FINAL_ONLY

class SNESConvergedReason(object):
    # iterating
    CONVERGED_ITERATING      = SNES_CONVERGED_ITERATING
    ITERATING                = SNES_CONVERGED_ITERATING
    # converged
    CONVERGED_FNORM_ABS      = SNES_CONVERGED_FNORM_ABS
    CONVERGED_FNORM_RELATIVE = SNES_CONVERGED_FNORM_RELATIVE
    CONVERGED_SNORM_RELATIVE = SNES_CONVERGED_SNORM_RELATIVE
    CONVERGED_ITS            = SNES_CONVERGED_ITS
    # diverged
    DIVERGED_FUNCTION_DOMAIN = SNES_DIVERGED_FUNCTION_DOMAIN
    DIVERGED_FUNCTION_COUNT  = SNES_DIVERGED_FUNCTION_COUNT
    DIVERGED_LINEAR_SOLVE    = SNES_DIVERGED_LINEAR_SOLVE
    DIVERGED_FNORM_NAN       = SNES_DIVERGED_FNORM_NAN
    DIVERGED_MAX_IT          = SNES_DIVERGED_MAX_IT
    DIVERGED_LINE_SEARCH     = SNES_DIVERGED_LINE_SEARCH
    DIVERGED_INNER           = SNES_DIVERGED_INNER
    DIVERGED_LOCAL_MIN       = SNES_DIVERGED_LOCAL_MIN
    DIVERGED_DTOL            = SNES_DIVERGED_DTOL
    DIVERGED_JACOBIAN_DOMAIN = SNES_DIVERGED_JACOBIAN_DOMAIN
    DIVERGED_TR_DELTA        = SNES_DIVERGED_TR_DELTA

# --------------------------------------------------------------------

cdef class SNES(Object):

    Type = SNESType
    NormSchedule = SNESNormSchedule
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
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( SNESSetOptionsPrefix(self.snes, cval) )

    def getOptionsPrefix(self):
        cdef const char *cval = NULL
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

    def setDM(self, DM dm):
        CHKERR( SNESSetDM(self.snes, dm.dm) )

    # --- FAS ---
    def setFASInterpolation(self, level, Mat mat):
        cdef PetscInt clevel = asInt(level)
        CHKERR( SNESFASSetInterpolation(self.snes, clevel, mat.mat) )

    def getFASInterpolation(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef Mat mat = Mat()
        CHKERR( SNESFASGetInterpolation(self.snes, clevel, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

    def setFASRestriction(self, level, Mat mat):
        cdef PetscInt clevel = asInt(level)
        CHKERR( SNESFASSetRestriction(self.snes, clevel, mat.mat) )

    def getFASRestriction(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef Mat mat = Mat()
        CHKERR( SNESFASGetRestriction(self.snes, clevel, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

    def setFASInjection(self, level, Mat mat):
        cdef PetscInt clevel = asInt(level)
        CHKERR( SNESFASSetInjection(self.snes, clevel, mat.mat) )

    def getFASInjection(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef Mat mat = Mat()
        CHKERR( SNESFASGetInjection(self.snes, clevel, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

    def setFASRScale(self, level, Vec vec):
        cdef PetscInt clevel = asInt(level)
        CHKERR( SNESFASSetRScale(self.snes, clevel, vec.vec) )

    def setFASLevels(self, levels, comms=None):
        cdef PetscInt clevels = asInt(levels)
        cdef MPI_Comm *ccomms = NULL
        cdef Py_ssize_t i = 0
        if comms is not None:
            if clevels != <PetscInt>len(comms):
                raise ValueError("Must provide as many communicators as levels")
            CHKERR( PetscMalloc(sizeof(MPI_Comm)*<size_t>clevels, &ccomms) )
            try:
                for i, comm in enumerate(comms):
                    ccomms[i] = def_Comm(comm, MPI_COMM_NULL)
                CHKERR( SNESFASSetLevels(self.snes, clevels, ccomms) )
            finally:
                CHKERR( PetscFree(ccomms) )
        else:
            CHKERR( SNESFASSetLevels(self.snes, clevels, ccomms) )

    def getFASLevels(self):
        cdef PetscInt levels = 0
        CHKERR( SNESFASGetLevels(self.snes, &levels) )
        return toInt(levels)

    def getFASCycleSNES(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef SNES lsnes = SNES()
        CHKERR( SNESFASGetCycleSNES(self.snes, clevel, &lsnes.snes) )
        PetscINCREF(lsnes.obj)
        return lsnes

    def getFASCoarseSolve(self):
        cdef SNES smooth = SNES()
        CHKERR( SNESFASGetCoarseSolve(self.snes, &smooth.snes) )
        PetscINCREF(smooth.obj)
        return smooth

    def getFASSmoother(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef SNES smooth = SNES()
        CHKERR( SNESFASGetSmoother(self.snes, clevel, &smooth.snes) )
        PetscINCREF(smooth.obj)
        return smooth

    def getFASSmootherDown(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef SNES smooth = SNES()
        CHKERR( SNESFASGetSmootherDown(self.snes, clevel, &smooth.snes) )
        PetscINCREF(smooth.obj)
        return smooth

    def getFASSmootherUp(self, level):
        cdef PetscInt clevel = asInt(level)
        cdef SNES smooth = SNES()
        CHKERR( SNESFASGetSmootherUp(self.snes, clevel, &smooth.snes) )
        PetscINCREF(smooth.obj)
        return smooth
    # --- nonlinear preconditioner ---

    def getNPC(self):
        cdef SNES snes = SNES()
        CHKERR( SNESGetNPC(self.snes, &snes.snes) )
        PetscINCREF(snes.obj)
        return snes

    def hasNPC(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( SNESHasNPC(self.snes, &flag) )
        return toBool(flag)

    def setNPC(self, SNES snes):
        CHKERR( SNESSetNPC(self.snes, snes.snes) )

    # --- user Function/Jacobian routines ---

    def setLineSearchPreCheck(self, precheck, args=None, kargs=None):
        cdef PetscSNESLineSearch snesls = NULL
        SNESGetLineSearch(self.snes, &snesls)
        if precheck is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (precheck, args, kargs)
            self.set_attr('__precheck__', context)
            CHKERR( SNESLineSearchSetPreCheck(snesls, SNES_PreCheck, <void*> context) )
        else:
            self.set_attr('__precheck__', None)
            CHKERR( SNESLineSearchSetPreCheck(snesls, NULL, NULL) )

    def setInitialGuess(self, initialguess, args=None, kargs=None):
        if initialguess is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (initialguess, args, kargs)
            self.set_attr('__initialguess__', context)
            CHKERR( SNESSetInitialGuess(self.snes, SNES_InitialGuess, <void*>context) )
        else:
            self.set_attr('__initialguess__', None)
            CHKERR( SNESSetInitialGuess(self.snes, NULL, NULL) )

    def getInitialGuess(self):
        return self.get_attr('__initialguess__')

    def setFunction(self, function, Vec f, args=None, kargs=None):
        cdef PetscVec fvec=NULL
        if f is not None: fvec = f.vec
        if function is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (function, args, kargs)
            self.set_attr('__function__', context)
            CHKERR( SNESSetFunction(self.snes, fvec, SNES_Function, <void*>context) )
        else:
            CHKERR( SNESSetFunction(self.snes, fvec, NULL, NULL) )

    def getFunction(self):
        cdef Vec f = Vec()
        cdef void* ctx
        cdef int (*fun)(PetscSNES,PetscVec,PetscVec,void*)
        CHKERR( SNESGetFunction(self.snes, &f.vec, &fun, &ctx) )
        PetscINCREF(f.obj)
        cdef object function = self.get_attr('__function__')
        cdef object context

        if function is not None:
            return (f, function)

        if ctx != NULL and <void*>SNES_Function == <void*>fun:
            context = <object>ctx
            if context is not None:
                assert type(context) is tuple
                return (f, context)

        return (f, None)

    def setUpdate(self, update, args=None, kargs=None):
        if update is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (update, args, kargs)
            self.set_attr('__update__', context)
            CHKERR( SNESSetUpdate(self.snes, SNES_Update) )
        else:
            self.set_attr('__update__', None)
            CHKERR( SNESSetUpdate(self.snes, NULL) )

    def getUpdate(self):
        return self.get_attr('__update__')

    def setJacobian(self, jacobian, Mat J=None, Mat P=None, args=None, kargs=None):
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat=Jmat
        if P is not None: Pmat = P.mat
        if jacobian is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (jacobian, args, kargs)
            self.set_attr('__jacobian__', context)
            CHKERR( SNESSetJacobian(self.snes, Jmat, Pmat, SNES_Jacobian, <void*>context) )
        else:
            CHKERR( SNESSetJacobian(self.snes, Jmat, Pmat, NULL, NULL) )

    def getJacobian(self):
        cdef Mat J = Mat()
        cdef Mat P = Mat()
        CHKERR( SNESGetJacobian(self.snes, &J.mat, &P.mat, NULL, NULL) )
        PetscINCREF(J.obj)
        PetscINCREF(P.obj)
        cdef object jacobian = self.get_attr('__jacobian__')
        return (J, P, jacobian)

    def setObjective(self, objective, args=None, kargs=None):
        if objective is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (objective, args, kargs)
            self.set_attr('__objective__', context)
            CHKERR( SNESSetObjective(self.snes, SNES_Objective, <void*>context) )
        else:
            CHKERR( SNESSetObjective(self.snes, NULL, NULL) )

    def getObjective(self):
        CHKERR( SNESGetObjective(self.snes, NULL, NULL) )
        cdef object objective = self.get_attr('__objective__')
        return objective

    def computeFunction(self, Vec x, Vec f):
        CHKERR( SNESComputeFunction(self.snes, x.vec, f.vec) )

    def computeJacobian(self, Vec x, Mat J, Mat P=None):
        cdef PetscMat jmat = J.mat, pmat = J.mat
        if P is not None: pmat = P.mat
        CHKERR( SNESComputeJacobian(self.snes, x.vec, jmat, pmat) )

    def computeObjective(self, Vec x):
        cdef PetscReal o = 0
        CHKERR( SNESComputeObjective(self.snes, x.vec, &o) )
        return toReal(o)

    def setNGS(self, ngs, args=None, kargs=None):
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (ngs, args, kargs)
        self.set_attr('__ngs__', context)
        CHKERR( SNESSetNGS(self.snes, SNES_NGS, <void*>context) )

    def getNGS(self):
        CHKERR( SNESGetNGS(self.snes, NULL, NULL) )
        cdef object ngs = self.get_attr('__ngs__')
        return ngs

    def computeNGS(self, Vec x, Vec b=None):
        cdef PetscVec bvec = NULL
        if b is not None: bvec = b.vec
        CHKERR( SNESComputeNGS(self.snes, bvec, x.vec) )

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

    def setNormSchedule(self, normsched):
        CHKERR( SNESSetNormSchedule(self.snes, normsched) )

    def getNormSchedule(self):
        cdef PetscSNESNormSchedule normsched = SNES_NORM_NONE
        CHKERR( SNESGetNormSchedule(self.snes, &normsched) )
        return normsched

    def setConvergenceTest(self, converged, args=None, kargs=None):
        if converged == "skip":
            self.set_attr('__converged__', None)
            CHKERR( SNESSetConvergenceTest(self.snes, SNESConvergedSkip, NULL, NULL) )
        elif converged is None or converged == "default":
            self.set_attr('__converged__', None)
            CHKERR( SNESSetConvergenceTest(self.snes, SNESConvergedDefault, NULL, NULL) )
        else:
            assert callable(converged)
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (converged, args, kargs)
            self.set_attr('__converged__', context)
            CHKERR( SNESSetConvergenceTest(self.snes, SNES_Converged, <void*>context, NULL) )

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

    def logConvergenceHistory(self, norm, linear_its=0):
        cdef PetscReal rval = asReal(norm)
        cdef PetscInt  ival = asInt(linear_its)
        CHKERR( SNESLogConvergenceHistory(self.snes, rval, ival) )

    def setResetCounters(self, reset=True):
        cdef PetscBool flag = reset
        CHKERR( SNESSetCountersReset(self.snes, flag) )

    # --- monitoring ---

    def setMonitor(self, monitor, args=None, kargs=None):
        if monitor is None: return
        cdef object monitorlist = self.get_attr('__monitor__')
        if monitorlist is None:
            monitorlist = []
            self.set_attr('__monitor__', monitorlist)
            CHKERR( SNESMonitorSet(self.snes, SNES_Monitor, NULL, NULL) )
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (monitor, args, kargs)
        monitorlist.append(context)

    def getMonitor(self):
        return self.get_attr('__monitor__')

    def monitorCancel(self):
        CHKERR( SNESMonitorCancel(self.snes) )
        self.set_attr('__monitor__', None)

    cancelMonitor = monitorCancel

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

    def setMaxStepFailures(self, max_fails):
        cdef PetscInt ival = asInt(max_fails)
        CHKERR( SNESSetMaxNonlinearStepFailures(self.snes, ival) )

    def getMaxStepFailures(self):
        cdef PetscInt ival = 0
        CHKERR( SNESGetMaxNonlinearStepFailures(self.snes, &ival) )
        return toInt(ival)

    def getStepFailures(self):
        cdef PetscInt ival = 0
        CHKERR( SNESGetNonlinearStepFailures(self.snes, &ival) )
        return toInt(ival)

    def setMaxKSPFailures(self, max_fails):
        cdef PetscInt ival = asInt(max_fails)
        CHKERR( SNESSetMaxLinearSolveFailures(self.snes, ival) )

    def getMaxKSPFailures(self):
        cdef PetscInt ival = 0
        CHKERR( SNESGetMaxLinearSolveFailures(self.snes, &ival) )
        return toInt(ival)

    def getKSPFailures(self):
        cdef PetscInt ival = 0
        CHKERR( SNESGetLinearSolveFailures(self.snes, &ival) )
        return toInt(ival)

    setMaxNonlinearStepFailures = setMaxStepFailures
    getMaxNonlinearStepFailures = getMaxStepFailures
    getNonlinearStepFailures    = getStepFailures
    setMaxLinearSolveFailures   = setMaxKSPFailures
    getMaxLinearSolveFailures   = getMaxKSPFailures
    getLinearSolveFailures      = getKSPFailures

    # --- solving ---

    def setUp(self):
        CHKERR( SNESSetUp(self.snes) )

    def reset(self):
        CHKERR( SNESReset(self.snes) )

    def solve(self, Vec b or None, Vec x):
        cdef PetscVec rhs = NULL
        if b is not None: rhs = b.vec
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

    def setSolution(self, Vec vec):
        CHKERR( SNESSetSolution(self.snes, vec.vec) )

    def getSolutionUpdate(self):
        cdef Vec vec = Vec()
        CHKERR( SNESGetSolutionUpdate(self.snes, &vec.vec) )
        PetscINCREF(vec.obj)
        return vec

    # --- linear solver ---

    def setKSP(self, KSP ksp):
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
        return toBool(flag)

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
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( SNESGetUseMFFD(self.snes, &flag) )
        return toBool(flag)

    def setUseFD(self, flag=True):
        cdef PetscBool bval = flag
        CHKERR( SNESSetUseFDColoring(self.snes, bval) )

    def getUseFD(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( SNESGetUseFDColoring(self.snes, &flag) )
        return toBool(flag)

    # --- VI ---

    def setVariableBounds(self, Vec xl, Vec xu):
        CHKERR( SNESVISetVariableBounds(self.snes, xl.vec, xu.vec) )

    def getVIInactiveSet(self):
        cdef IS inact = IS()
        CHKERR( SNESVIGetInactiveSet(self.snes, &inact.iset) )
        PetscINCREF(inact.obj)
        return inact

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
        cdef const char *cval = NULL
        py_type = str2bytes(py_type, &cval)
        CHKERR( SNESPythonSetType(self.snes, cval) )

    # --- Composite ---

    def getCompositeSNES(self, n):
        cdef PetscInt cn
        cdef SNES snes = SNES()
        cn = asInt(n)
        CHKERR( SNESCompositeGetSNES(self.snes, cn, &snes.snes) )
        PetscINCREF(snes.obj)
        return snes

    def getCompositeNumber(self):
        cdef PetscInt cn = 0
        CHKERR( SNESCompositeGetNumber(self.snes, &cn) )
        return toInt(cn)

    # --- NASM ---

    def getNASMSNES(self, n):
        cdef PetscInt cn = asInt(n)
        cdef SNES snes = SNES()
        CHKERR( SNESNASMGetSNES(self.snes, cn, &snes.snes) )
        PetscINCREF(snes.obj)
        return snes

    def getNASMNumber(self):
        cdef PetscInt cn = 0
        CHKERR( SNESNASMGetNumber(self.snes, &cn) )
        return toInt(cn)

    # --- Patch ---

    def setPatchCellNumbering(self, Section sec not None):
        CHKERR( SNESPatchSetCellNumbering(self.snes, sec.sec) )

    def setPatchDiscretisationInfo(self, dms, bs,
                                   cellNodeMaps,
                                   subspaceOffsets,
                                   ghostBcNodes,
                                   globalBcNodes):
        cdef PetscInt numSubSpaces = 0
        cdef PetscInt numGhostBcs = 0, numGlobalBcs = 0
        cdef PetscInt *nodesPerCell = NULL
        cdef const PetscInt **ccellNodeMaps = NULL
        cdef PetscDM *cdms = NULL
        cdef PetscInt *cbs = NULL
        cdef PetscInt *csubspaceOffsets = NULL
        cdef PetscInt *cghostBcNodes = NULL
        cdef PetscInt *cglobalBcNodes = NULL
        cdef PetscInt i = 0

        bs = iarray_i(bs, &numSubSpaces, &cbs)
        ghostBcNodes = iarray_i(ghostBcNodes, &numGhostBcs, &cghostBcNodes)
        globalBcNodes = iarray_i(globalBcNodes, &numGlobalBcs, &cglobalBcNodes)
        subspaceOffsets = iarray_i(subspaceOffsets, NULL, &csubspaceOffsets)

        CHKERR( PetscMalloc(<size_t>numSubSpaces*sizeof(PetscInt), &nodesPerCell) )
        CHKERR( PetscMalloc(<size_t>numSubSpaces*sizeof(PetscDM), &cdms) )
        CHKERR( PetscMalloc(<size_t>numSubSpaces*sizeof(PetscInt*), &ccellNodeMaps) )
        for i in range(numSubSpaces):
            cdms[i] = (<DM?>dms[i]).dm
            _, nodes = asarray(cellNodeMaps[i]).shape
            cellNodeMaps[i] = iarray_i(cellNodeMaps[i], NULL, <PetscInt**>&(ccellNodeMaps[i]))
            nodesPerCell[i] = asInt(nodes)

        # TODO: refactor on the PETSc side to take ISes?
        CHKERR( SNESPatchSetDiscretisationInfo(self.snes, numSubSpaces,
                                               cdms, cbs, nodesPerCell,
                                               ccellNodeMaps, csubspaceOffsets,
                                               numGhostBcs, cghostBcNodes,
                                               numGlobalBcs, cglobalBcNodes) )
        CHKERR( PetscFree(nodesPerCell) )
        CHKERR( PetscFree(cdms) )
        CHKERR( PetscFree(ccellNodeMaps) )

    def setPatchComputeOperator(self, operator, args=None, kargs=None):
        if args is  None: args  = ()
        if kargs is None: kargs = {}
        context = (operator, args, kargs)
        self.set_attr("__patch_compute_operator__", context)
        CHKERR( SNESPatchSetComputeOperator(self.snes, PCPatch_ComputeOperator, <void*>context) )

    def setPatchComputeFunction(self, function, args=None, kargs=None):
        if args is  None: args  = ()
        if kargs is None: kargs = {}
        context = (function, args, kargs)
        self.set_attr("__patch_compute_function__", context)
        CHKERR( SNESPatchSetComputeFunction(self.snes, PCPatch_ComputeFunction, <void*>context) )

    def setPatchConstructType(self, typ, operator=None, args=None, kargs=None):
        if args is  None: args  = ()
        if kargs is None: kargs = {}

        if typ in {PC.PatchConstructType.PYTHON, PC.PatchConstructType.USER} and operator is None:
            raise ValueError("Must provide operator for USER or PYTHON type")
        if operator is not None:
            context = (operator, args, kargs)
        else:
            context = None
        self.set_attr("__patch_construction_operator__", context)
        CHKERR( SNESPatchSetConstructType(self.snes, typ, PCPatch_UserConstructOperator, <void*>context) )

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

    # --- nonlinear preconditioner ---

    property npc:
        def __get__(self):
            return self.getNPC()
        def __set__(self, value):
            self.setNPC(value)

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
del SNESNormSchedule
del SNESConvergedReason

# --------------------------------------------------------------------
