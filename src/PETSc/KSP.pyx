# --------------------------------------------------------------------

class KSPType(object):
    RICHARDSON = S_(KSPRICHARDSON)
    CHEBYSHEV  = S_(KSPCHEBYSHEV)
    CG         = S_(KSPCG)
    GROPPCG    = S_(KSPGROPPCG)
    PIPECG     = S_(KSPPIPECG)
    PIPECGRR   = S_(KSPPIPECGRR)
    PIPELCG    = S_(KSPPIPELCG)
    PIPEPRCG   = S_(KSPPIPEPRCG)
    CGNE       = S_(KSPCGNE)
    NASH       = S_(KSPNASH)
    STCG       = S_(KSPSTCG)
    GLTR       = S_(KSPGLTR)
    FCG        = S_(KSPFCG)
    PIPEFCG    = S_(KSPPIPEFCG)
    GMRES      = S_(KSPGMRES)
    PIPEFGMRES = S_(KSPPIPEFGMRES)
    FGMRES     = S_(KSPFGMRES)
    LGMRES     = S_(KSPLGMRES)
    DGMRES     = S_(KSPDGMRES)
    PGMRES     = S_(KSPPGMRES)
    TCQMR      = S_(KSPTCQMR)
    BCGS       = S_(KSPBCGS)
    IBCGS      = S_(KSPIBCGS)
    FBCGS      = S_(KSPFBCGS)
    FBCGSR     = S_(KSPFBCGSR)
    BCGSL      = S_(KSPBCGSL)
    PIPEBCGS   = S_(KSPPIPEBCGS)
    CGS        = S_(KSPCGS)
    TFQMR      = S_(KSPTFQMR)
    CR         = S_(KSPCR)
    PIPECR     = S_(KSPPIPECR)
    LSQR       = S_(KSPLSQR)
    PREONLY    = S_(KSPPREONLY)
    QCG        = S_(KSPQCG)
    BICG       = S_(KSPBICG)
    MINRES     = S_(KSPMINRES)
    SYMMLQ     = S_(KSPSYMMLQ)
    LCD        = S_(KSPLCD)
    PYTHON     = S_(KSPPYTHON)
    GCR        = S_(KSPGCR)
    PIPEGCR    = S_(KSPPIPEGCR)
    TSIRM      = S_(KSPTSIRM)
    CGLS       = S_(KSPCGLS)
    FETIDP     = S_(KSPFETIDP)
    HPDDM      = S_(KSPHPDDM)

class KSPNormType(object):
    # native
    NORM_DEFAULT          = KSP_NORM_DEFAULT
    NORM_NONE             = KSP_NORM_NONE
    NORM_PRECONDITIONED   = KSP_NORM_PRECONDITIONED
    NORM_UNPRECONDITIONED = KSP_NORM_UNPRECONDITIONED
    NORM_NATURAL          = KSP_NORM_NATURAL
    # aliases
    DEFAULT          = NORM_DEFAULT
    NONE = NO        = NORM_NONE
    PRECONDITIONED   = NORM_PRECONDITIONED
    UNPRECONDITIONED = NORM_UNPRECONDITIONED
    NATURAL          = NORM_NATURAL

class KSPConvergedReason(object):
    #iterating
    CONVERGED_ITERATING       = KSP_CONVERGED_ITERATING
    ITERATING                 = KSP_CONVERGED_ITERATING
    # converged
    CONVERGED_RTOL_NORMAL     = KSP_CONVERGED_RTOL_NORMAL
    CONVERGED_ATOL_NORMAL     = KSP_CONVERGED_ATOL_NORMAL
    CONVERGED_RTOL            = KSP_CONVERGED_RTOL
    CONVERGED_ATOL            = KSP_CONVERGED_ATOL
    CONVERGED_ITS             = KSP_CONVERGED_ITS
    CONVERGED_CG_NEG_CURVE    = KSP_CONVERGED_CG_NEG_CURVE
    CONVERGED_CG_CONSTRAINED  = KSP_CONVERGED_CG_CONSTRAINED
    CONVERGED_STEP_LENGTH     = KSP_CONVERGED_STEP_LENGTH
    CONVERGED_HAPPY_BREAKDOWN = KSP_CONVERGED_HAPPY_BREAKDOWN
    # diverged
    DIVERGED_NULL             = KSP_DIVERGED_NULL
    DIVERGED_MAX_IT           = KSP_DIVERGED_MAX_IT
    DIVERGED_DTOL             = KSP_DIVERGED_DTOL
    DIVERGED_BREAKDOWN        = KSP_DIVERGED_BREAKDOWN
    DIVERGED_BREAKDOWN_BICG   = KSP_DIVERGED_BREAKDOWN_BICG
    DIVERGED_NONSYMMETRIC     = KSP_DIVERGED_NONSYMMETRIC
    DIVERGED_INDEFINITE_PC    = KSP_DIVERGED_INDEFINITE_PC
    DIVERGED_NANORINF         = KSP_DIVERGED_NANORINF
    DIVERGED_INDEFINITE_MAT   = KSP_DIVERGED_INDEFINITE_MAT
    DIVERGED_PCSETUP_FAILED   = KSP_DIVERGED_PC_FAILED

# --------------------------------------------------------------------

cdef class KSP(Object):

    Type            = KSPType
    NormType        = KSPNormType
    ConvergedReason = KSPConvergedReason

    # --- xxx ---

    def __cinit__(self):
        self.obj = <PetscObject*> &self.ksp
        self.ksp = NULL

    def __call__(self, b, x=None):
        if x is None: # XXX do this better
            x = self.getOperators()[0].createVecLeft()
        self.solve(b, x)
        return x

    # --- xxx ---

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( KSPView(self.ksp, vwr) )

    def destroy(self):
        CHKERR( KSPDestroy(&self.ksp) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscKSP newksp = NULL
        CHKERR( KSPCreate(ccomm, &newksp) )
        PetscCLEAR(self.obj); self.ksp = newksp
        return self

    def setType(self, ksp_type):
        cdef PetscKSPType cval = NULL
        ksp_type = str2bytes(ksp_type, &cval)
        CHKERR( KSPSetType(self.ksp, cval) )

    def getType(self):
        cdef PetscKSPType cval = NULL
        CHKERR( KSPGetType(self.ksp, &cval) )
        return bytes2str(cval)

    def setOptionsPrefix(self, prefix):
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( KSPSetOptionsPrefix(self.ksp, cval) )

    def getOptionsPrefix(self):
        cdef const char *cval = NULL
        CHKERR( KSPGetOptionsPrefix(self.ksp, &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( KSPSetFromOptions(self.ksp) )

    # --- application context ---

    def setAppCtx(self, appctx):
        self.set_attr('__appctx__', appctx)

    def getAppCtx(self):
        return self.get_attr('__appctx__')

    # --- discretization space ---

    def getDM(self):
        cdef PetscDM newdm = NULL
        CHKERR( KSPGetDM(self.ksp, &newdm) )
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        PetscINCREF(dm.obj)
        return dm

    def setDM(self, DM dm):
        CHKERR( KSPSetDM(self.ksp, dm.dm) )

    def setDMActive(self, bint flag):
        cdef PetscBool cflag = PETSC_FALSE
        if flag: cflag = PETSC_TRUE
        CHKERR( KSPSetDMActive(self.ksp, cflag) )

    # --- operators and preconditioner ---

    def setComputeRHS(self, rhs, args=None, kargs=None):
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (rhs, args, kargs)
        self.set_attr('__rhs__', context)
        CHKERR( KSPSetComputeRHS(self.ksp, KSP_ComputeRHS, <void*>context) )

    def setComputeOperators(self, operators, args=None, kargs=None):
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (operators, args, kargs)
        self.set_attr('__operators__', context)
        CHKERR( KSPSetComputeOperators(self.ksp, KSP_ComputeOps, <void*>context) )

    def setOperators(self, Mat A=None, Mat P=None):
        cdef PetscMat amat=NULL
        if A is not None: amat = A.mat
        cdef PetscMat pmat=amat
        if P is not None: pmat = P.mat
        CHKERR( KSPSetOperators(self.ksp, amat, pmat) )

    def getOperators(self):
        cdef Mat A = Mat(), P = Mat()
        CHKERR( KSPGetOperators(self.ksp, &A.mat, &P.mat) )
        PetscINCREF(A.obj)
        PetscINCREF(P.obj)
        return (A, P)

    def setPC(self, PC pc):
        CHKERR( KSPSetPC(self.ksp, pc.pc) )

    def getPC(self):
        cdef PC pc = PC()
        CHKERR( KSPGetPC(self.ksp, &pc.pc) )
        PetscINCREF(pc.obj)
        return pc

    # --- tolerances and convergence ---

    def setTolerances(self, rtol=None, atol=None, divtol=None, max_it=None):
        cdef PetscReal crtol, catol, cdivtol
        crtol = catol = cdivtol = PETSC_DEFAULT;
        if rtol   is not None: crtol   = asReal(rtol)
        if atol   is not None: catol   = asReal(atol)
        if divtol is not None: cdivtol = asReal(divtol)
        cdef PetscInt cmaxits = PETSC_DEFAULT
        if max_it is not None: cmaxits = asInt(max_it)
        CHKERR( KSPSetTolerances(self.ksp, crtol, catol, cdivtol, cmaxits) )

    def getTolerances(self):
        cdef PetscReal crtol=0, catol=0, cdivtol=0
        cdef PetscInt cmaxits=0
        CHKERR( KSPGetTolerances(self.ksp, &crtol, &catol, &cdivtol, &cmaxits) )
        return (toReal(crtol), toReal(catol), toReal(cdivtol), toInt(cmaxits))

    def setConvergenceTest(self, converged, args=None, kargs=None):
        cdef PetscKSPNormType normtype = KSP_NORM_NONE
        cdef void* cctx = NULL
        if converged is not None:
            CHKERR( KSPSetConvergenceTest(
                    self.ksp, KSP_Converged, NULL, NULL) )
            if args is None: args = ()
            if kargs is None: kargs = {}
            self.set_attr('__converged__', (converged, args, kargs))
        else:
            CHKERR( KSPGetNormType(self.ksp, &normtype) )
            if normtype != KSP_NORM_NONE:
                CHKERR( KSPConvergedDefaultCreate(&cctx) )
                CHKERR( KSPSetConvergenceTest(
                        self.ksp, KSPConvergedDefault,
                        cctx, KSPConvergedDefaultDestroy) )
            else:
                CHKERR( KSPSetConvergenceTest(
                        self.ksp, KSPConvergedSkip,
                        NULL, NULL) )
            self.set_attr('__converged__', None)

    def getConvergenceTest(self):
        return self.get_attr('__converged__')

    def callConvergenceTest(self, its, rnorm):
        cdef PetscInt  ival = asInt(its)
        cdef PetscReal rval = asReal(rnorm)
        cdef PetscKSPConvergedReason reason = KSP_CONVERGED_ITERATING
        CHKERR( KSPConvergenceTestCall(self.ksp, ival, rval, &reason) )
        return reason

    def setConvergenceHistory(self, length=None, reset=False):
        cdef PetscReal *data = NULL
        cdef PetscInt   size = 10000
        cdef PetscBool flag = PETSC_FALSE
        if   length is True:     pass
        elif length is not None: size = asInt(length)
        if size < 0: size = 10000
        if reset: flag = PETSC_TRUE
        cdef object hist = oarray_r(empty_r(size), NULL, &data)
        self.set_attr('__history__', hist)
        CHKERR( KSPSetResidualHistory(self.ksp, data, size, flag) )

    def getConvergenceHistory(self):
        cdef PetscReal *data = NULL
        cdef PetscInt   size = 0
        CHKERR( KSPGetResidualHistory(self.ksp, &data, &size) )
        return array_r(size, data)

    def logConvergenceHistory(self, rnorm):
        cdef PetscReal rval = asReal(rnorm)
        CHKERR( KSPLogResidualHistory(self.ksp, rval) )

    # --- monitoring ---

    def setMonitor(self, monitor, args=None, kargs=None):
        if monitor is None: return
        cdef object monitorlist = self.get_attr('__monitor__')
        if monitorlist is None:
            monitorlist = []
            self.set_attr('__monitor__', monitorlist)
            CHKERR( KSPMonitorSet(self.ksp, KSP_Monitor, NULL, NULL) )
        if args is None: args = ()
        if kargs is None: kargs = {}
        monitorlist.append((monitor, args, kargs))

    def getMonitor(self):
        return self.get_attr('__monitor__')

    def cancelMonitor(self):
        CHKERR( KSPMonitorCancel(self.ksp) )
        self.set_attr('__monitor__', None)

    def monitor(self, its, rnorm):
        cdef PetscInt  ival = asInt(its)
        cdef PetscReal rval = asReal(rnorm)
        CHKERR( KSPMonitor(self.ksp, ival, rval) )

    # --- customization ---

    def setPCSide(self, side):
        CHKERR( KSPSetPCSide(self.ksp, side) )

    def getPCSide(self):
        cdef PetscPCSide side = PC_LEFT
        CHKERR( KSPGetPCSide(self.ksp, &side) )
        return side

    def setNormType(self, normtype):
        CHKERR( KSPSetNormType(self.ksp, normtype) )

    def getNormType(self):
        cdef PetscKSPNormType normtype = KSP_NORM_NONE
        CHKERR( KSPGetNormType(self.ksp, &normtype) )
        return normtype

    def setComputeEigenvalues(self, bint flag):
        cdef PetscBool compute = PETSC_FALSE
        if flag: compute = PETSC_TRUE
        CHKERR( KSPSetComputeEigenvalues(self.ksp, compute) )

    def getComputeEigenvalues(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( KSPGetComputeEigenvalues(self.ksp, &flag) )
        return toBool(flag)

    def setComputeSingularValues(self, bint flag):
        cdef PetscBool compute = PETSC_FALSE
        if flag: compute = PETSC_TRUE
        CHKERR( KSPSetComputeSingularValues(self.ksp, compute) )

    def getComputeSingularValues(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( KSPGetComputeSingularValues(self.ksp, &flag) )
        return toBool(flag)

    # --- initial guess ---

    def setInitialGuessNonzero(self, bint flag):
        cdef PetscBool guess_nonzero = PETSC_FALSE
        if flag: guess_nonzero = PETSC_TRUE
        CHKERR( KSPSetInitialGuessNonzero(self.ksp, guess_nonzero) )

    def getInitialGuessNonzero(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( KSPGetInitialGuessNonzero(self.ksp, &flag) )
        return toBool(flag)

    def setInitialGuessKnoll(self, bint flag):
        cdef PetscBool guess_knoll = PETSC_FALSE
        if flag: guess_knoll = PETSC_TRUE
        CHKERR( KSPSetInitialGuessKnoll(self.ksp, guess_knoll) )

    def getInitialGuessKnoll(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( KSPGetInitialGuessKnoll(self.ksp, &flag) )
        return toBool(flag)

    def setUseFischerGuess(self, model, size):
        cdef PetscInt ival1 = asInt(model)
        cdef PetscInt ival2 = asInt(size)
        CHKERR( KSPSetUseFischerGuess(self.ksp, ival1, ival2) )

    # --- solving ---

    def setUp(self):
        CHKERR( KSPSetUp(self.ksp) )

    def reset(self):
        CHKERR( KSPReset(self.ksp) )

    def setUpOnBlocks(self):
        CHKERR( KSPSetUpOnBlocks(self.ksp) )

    def solve(self, Vec b or None, Vec x or None):
        cdef PetscVec b_vec = NULL
        cdef PetscVec x_vec = NULL
        if b is not None: b_vec = b.vec
        if x is not None: x_vec = x.vec
        CHKERR( KSPSolve(self.ksp, b_vec, x_vec) )

    def solveTranspose(self, Vec b, Vec x):
        CHKERR( KSPSolveTranspose(self.ksp, b.vec, x.vec) )

    def setIterationNumber(self, its):
        cdef PetscInt ival = asInt(its)
        CHKERR( KSPSetIterationNumber(self.ksp, ival) )

    def getIterationNumber(self):
        cdef PetscInt ival = 0
        CHKERR( KSPGetIterationNumber(self.ksp, &ival) )
        return toInt(ival)

    def setResidualNorm(self, rnorm):
        cdef PetscReal rval = asReal(rnorm)
        CHKERR( KSPSetResidualNorm(self.ksp, rval) )

    def getResidualNorm(self):
        cdef PetscReal rval = 0
        CHKERR( KSPGetResidualNorm(self.ksp, &rval) )
        return toReal(rval)

    def setConvergedReason(self, reason):
        cdef PetscKSPConvergedReason val = reason
        CHKERR( KSPSetConvergedReason(self.ksp, val) )

    def getConvergedReason(self):
        cdef PetscKSPConvergedReason reason = KSP_CONVERGED_ITERATING
        CHKERR( KSPGetConvergedReason(self.ksp, &reason) )
        return reason

    def getRhs(self):
        cdef Vec vec = Vec()
        CHKERR( KSPGetRhs(self.ksp, &vec.vec) )
        PetscINCREF(vec.obj)
        return vec

    def getSolution(self):
        cdef Vec vec = Vec()
        CHKERR( KSPGetSolution(self.ksp, &vec.vec) )
        PetscINCREF(vec.obj)
        return vec

    def getWorkVecs(self, right=None, left=None):
        cdef bint R = right is not None
        cdef bint L = left  is not None
        cdef PetscInt i=0, nr=0, nl=0
        cdef PetscVec *vr=NULL, *vl=NULL
        if R: nr = asInt(right)
        if L: nl = asInt(left)
        cdef object vecsr = [] if R else None
        cdef object vecsl = [] if L else None
        CHKERR( KSPCreateVecs(self.ksp, nr, &vr, nl, &vr) )
        try:
            for i from 0 <= i < nr:
                vecsr.append(ref_Vec(vr[i]))
            for i from 0 <= i < nl:
                vecsl.append(ref_Vec(vl[i]))
        finally:
            if nr > 0 and vr != NULL:
                VecDestroyVecs(nr, &vr) # XXX errors?
            if nl > 0 and vl !=NULL:
                VecDestroyVecs(nl, &vl) # XXX errors?
        #
        if R and L: return (vecsr, vecsl)
        elif R:     return vecsr
        elif L:     return vecsl
        else:       return None

    def buildSolution(self, Vec x=None):
        if x is None: x = Vec()
        if x.vec == NULL:
            CHKERR( KSPGetSolution(self.ksp, &x.vec) )
            CHKERR( VecDuplicate(x.vec, &x.vec) )
        CHKERR( KSPBuildSolution(self.ksp, x.vec, NULL) )
        return x

    def buildResidual(self, Vec r=None):
        if r is None: r = Vec()
        if r.vec == NULL:
            CHKERR( KSPGetRhs(self.ksp, &r.vec) )
            CHKERR( VecDuplicate(r.vec, &r.vec) )
        CHKERR( KSPBuildResidual(self.ksp , NULL, r.vec, &r.vec) )
        return r

    def computeEigenvalues(self):
        cdef PetscInt its = 0
        cdef PetscInt neig = 0
        cdef PetscReal *rdata = NULL
        cdef PetscReal *idata = NULL
        CHKERR( KSPGetIterationNumber(self.ksp, &its) )
        cdef ndarray r = oarray_r(empty_r(its), NULL, &rdata)
        cdef ndarray i = oarray_r(empty_r(its), NULL, &idata)
        CHKERR( KSPComputeEigenvalues(self.ksp, its, rdata, idata, &neig) )
        eigen = empty_c(neig)
        eigen.real = r[:neig]
        eigen.imag = i[:neig]
        return eigen

    def computeExtremeSingularValues(self):
        cdef PetscReal smax = 0
        cdef PetscReal smin = 0
        CHKERR( KSPComputeExtremeSingularValues(self.ksp, &smax, &smin) )
        return smax, smin

    # --- GMRES ---

    def setGMRESRestart(self, restart):
        cdef PetscInt ival = asInt(restart)
        CHKERR( KSPGMRESSetRestart(self.ksp, ival) )

    # --- Python ---

    def createPython(self, context=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscKSP newksp = NULL
        CHKERR( KSPCreate(ccomm, &newksp) )
        PetscCLEAR(self.obj); self.ksp = newksp
        CHKERR( KSPSetType(self.ksp, KSPPYTHON) )
        CHKERR( KSPPythonSetContext(self.ksp, <void*>context) )
        return self

    def setPythonContext(self, context):
        CHKERR( KSPPythonSetContext(self.ksp, <void*>context) )

    def getPythonContext(self):
        cdef void *context = NULL
        CHKERR( KSPPythonGetContext(self.ksp, &context) )
        if context == NULL: return None
        else: return <object> context

    def setPythonType(self, py_type):
        cdef const char *cval = NULL
        py_type = str2bytes(py_type, &cval)
        CHKERR( KSPPythonSetType(self.ksp, cval) )

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

    property vec_rhs:
        def __get__(self):
            return self.getRhs()

    # --- operators ---

    property mat_op:
        def __get__(self):
            return self.getOperators()[0]

    property mat_pc:
        def __get__(self):
            return self.getOperators()[1]

    # --- initial guess ---

    property guess_nonzero:
        def __get__(self):
            return self.getInitialGuessNonzero()
        def __set__(self, value):
            self.setInitialGuessNonzero(value)

    property guess_knoll:
        def __get__(self):
            return self.getInitialGuessKnoll()
        def __set__(self, value):
            self.setInitialGuessKnoll(value)

    # --- preconditioner ---

    property pc:
        def __get__(self):
            return self.getPC()

    property pc_side:
        def __get__(self):
            return self.getPCSide()
        def __set__(self, value):
            self.setPCSide(value)

    property norm_type:
        def __get__(self):
            return self.getNormType()
        def __set__(self, value):
            self.setNormType(value)

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

    property divtol:
        def __get__(self):
            return self.getTolerances()[2]
        def __set__(self, value):
            self.setTolerances(divtol=value)

    property max_it:
        def __get__(self):
            return self.getTolerances()[3]
        def __set__(self, value):
            self.setTolerances(max_it=value)

    # --- iteration ---

    property its:
        def __get__(self):
            return self.getIterationNumber()
        def __set__(self, value):
            self.setIterationNumber(value)

    property norm:
        def __get__(self):
            return self.getResidualNorm()
        def __set__(self, value):
            self.setResidualNorm(value)

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

# --------------------------------------------------------------------

del KSPType
del KSPNormType
del KSPConvergedReason

# --------------------------------------------------------------------
