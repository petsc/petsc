# --------------------------------------------------------------------

class KSPType(object):
    RICHARDSON = KSPRICHARDSON
    CHEBYCHEV  = KSPCHEBYCHEV
    CG         = KSPCG
    CGNE       = KSPCGNE
    STCG       = KSPSTCG
    GLTR       = KSPGLTR
    GMRES      = KSPGMRES
    FGMRES     = KSPFGMRES
    LGMRES     = KSPLGMRES
    TCQMR      = KSPTCQMR
    BCGS       = KSPBCGS
    BCGSL      = KSPBCGSL
    CGS        = KSPCGS
    TFQMR      = KSPTFQMR
    CR         = KSPCR
    LSQR       = KSPLSQR
    PREONLY    = KSPPREONLY
    QCG        = KSPQCG
    BICG       = KSPBICG
    MINRES     = KSPMINRES
    SYMMLQ     = KSPSYMMLQ
    LCD        = KSPLCD
    #
    PYTHON = KSPPYTHON

class KSPNormType(object):
    # native
    NORM_NO               = KSP_NORM_NO
    NORM_PRECONDITIONED   = KSP_NORM_PRECONDITIONED
    NORM_UNPRECONDITIONED = KSP_NORM_UNPRECONDITIONED
    NORM_NATURAL          = KSP_NORM_NATURAL
    # aliases
    NONE = NO        = NORM_NO
    PRECONDITIONED   = NORM_PRECONDITIONED
    UNPRECONDITIONED = NORM_UNPRECONDITIONED
    NATURAL          = NORM_NATURAL

class KSPConvergedReason(object):
    #iterating
    CONVERGED_ITERATING       = KSP_CONVERGED_ITERATING
    ITERATING                 = KSP_CONVERGED_ITERATING
    # converged
    CONVERGED_RTOL            = KSP_CONVERGED_RTOL
    CONVERGED_ATOL            = KSP_CONVERGED_ATOL
    CONVERGED_ITS             = KSP_CONVERGED_ITS
    CONVERGED_CG_NEG_CURVE    = KSP_CONVERGED_CG_NEG_CURVE
    CONVERGED_CG_CONSTRAINED  = KSP_CONVERGED_CG_CONSTRAINED
    CONVERGED_STEP_LENGTH     = KSP_CONVERGED_STEP_LENGTH
    # diverged
    DIVERGED_NULL             = KSP_DIVERGED_NULL
    DIVERGED_MAX_IT           = KSP_DIVERGED_MAX_IT
    DIVERGED_DTOL             = KSP_DIVERGED_DTOL
    DIVERGED_BREAKDOWN        = KSP_DIVERGED_BREAKDOWN
    DIVERGED_BREAKDOWN_BICG   = KSP_DIVERGED_BREAKDOWN_BICG
    DIVERGED_NONSYMMETRIC     = KSP_DIVERGED_NONSYMMETRIC
    DIVERGED_INDEFINITE_PC    = KSP_DIVERGED_INDEFINITE_PC
    DIVERGED_NAN              = KSP_DIVERGED_NAN
    DIVERGED_INDEFINITE_MAT   = KSP_DIVERGED_INDEFINITE_MAT

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
            x = self.getOperators()[0].getVecLeft()
        self.solve(b, x)
        return x

    # --- xxx ---

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( KSPView(self.ksp, vwr) )

    def destroy(self):
        CHKERR( KSPDestroy(self.ksp) )
        self.ksp = NULL
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_WORLD)
        cdef PetscKSP newksp = NULL
        CHKERR( KSPCreate(ccomm, &newksp) )
        PetscCLEAR(self.obj); self.ksp = newksp
        return self

    def setType(self, ksp_type):
        CHKERR( KSPSetType(self.ksp, str2cp(ksp_type)) )

    def getType(self):
        cdef PetscKSPType ksp_type = NULL
        CHKERR( KSPGetType(self.ksp, &ksp_type) )
        return cp2str(ksp_type)

    def setOptionsPrefix(self, prefix):
        CHKERR( KSPSetOptionsPrefix(self.ksp, str2cp(prefix)) )

    def getOptionsPrefix(self):
        cdef const_char_p prefix = NULL
        CHKERR( KSPGetOptionsPrefix(self.ksp, &prefix) )
        return cp2str(prefix)

    def setFromOptions(self):
        CHKERR( KSPSetFromOptions(self.ksp) )

    # --- xxx ---

    def setAppCtx(self, appctx):
        Object_setAttr(<PetscObject>self.ksp, "__appctx__", appctx)

    def getAppCtx(self):
        return Object_getAttr(<PetscObject>self.ksp, "__appctx__")

    # --- xxx ---

    def setOperators(self, Mat A=None, Mat P=None, structure=None):
        cdef PetscMat amat=NULL
        if A is not None: amat = A.mat
        cdef PetscMat pmat=amat
        if P is not None: pmat = P.mat
        cdef PetscMatStructure flag = matstructure(structure)
        CHKERR( KSPSetOperators(self.ksp, amat, pmat, flag) )

    def getOperators(self):
        cdef Mat A = Mat(), P = Mat()
        cdef PetscMatStructure flag = MAT_DIFFERENT_NONZERO_PATTERN
        CHKERR( KSPGetOperators(self.ksp, &A.mat, &P.mat, &flag) )
        PetscIncref(<PetscObject>A.mat)
        PetscIncref(<PetscObject>P.mat)
        return (A, P, flag)

    def setNullSpace(self, NullSpace nsp not None):
        CHKERR( KSPSetNullSpace(self.ksp, nsp.nsp) )

    def getNullSpace(self):
        cdef NullSpace nsp = NullSpace()
        CHKERR( KSPGetNullSpace(self.ksp, &nsp.nsp) )
        PetscIncref(<PetscObject>nsp.nsp)
        return nsp

    def setPC(self, PC pc not None):
        CHKERR( KSPSetPC(self.ksp, pc.pc) )

    def getPC(self):
        cdef PC pc = PC()
        CHKERR( KSPGetPC(self.ksp, &pc.pc) )
        PetscIncref(<PetscObject>pc.pc)
        return pc

    def setPCSide(self, side):
        CHKERR( KSPSetPreconditionerSide(self.ksp, side) )

    def getPCSide(self):
        cdef PetscPCSide side = PC_LEFT
        CHKERR( KSPGetPreconditionerSide(self.ksp, &side) )
        return side

    def setNormType(self, normtype):
        CHKERR( KSPSetNormType(self.ksp, normtype) )

    def getNormType(self):
        cdef PetscKSPNormType normtype = KSP_NORM_NO
        CHKERR( KSPGetNormType(self.ksp, &normtype) )
        return normtype

    # --- xxx ---

    def setTolerances(self, rtol=None, atol=None, divtol=None, max_it=None):
        cdef PetscReal crtol, catol, cdivtol
        crtol = catol = cdivtol = PETSC_DEFAULT;
        if rtol   is not None: crtol   = rtol
        if atol   is not None: catol   = atol
        if divtol is not None: cdivtol = divtol
        cdef PetscInt cmaxits = PETSC_DEFAULT
        if max_it is not None: cmaxits = max_it
        CHKERR( KSPSetTolerances(self.ksp, crtol, catol, cdivtol, cmaxits) )

    def getTolerances(self):
        cdef PetscReal crtol, catol, cdivtol
        cdef PetscInt cmaxits
        CHKERR( KSPGetTolerances(self.ksp, &crtol, &catol, &cdivtol, &cmaxits) )
        return (crtol, catol, cdivtol, cmaxits)

    def setConvergenceTest(self, converged, *args, **kargs):
        if converged is None: KSP_setCnv(self.ksp, None)
        else: KSP_setCnv(self.ksp, (converged, args, kargs))

    def getConvergenceTest(self):
        return KSP_getCnv(self.ksp)
    
    def setConvergenceHistory(self, length=None, reset=False):
        cdef PetscReal *data = NULL
        cdef PetscInt   size = 10000
        cdef PetscTruth flag = PETSC_FALSE
        if   length is True:     pass
        elif length is not None: size = length
        if size < 0: size = 10000
        if reset: flag = PETSC_TRUE
        cdef ndarray hist = oarray_r(empty_r(size), NULL, &data)
        Object_setAttr(<PetscObject>self.ksp, "__history__", hist)
        CHKERR( KSPSetResidualHistory(self.ksp, data, size, flag) )

    def getConvergenceHistory(self):
        cdef PetscReal *data = NULL
        cdef PetscInt   size = 0
        CHKERR( KSPGetResidualHistory(self.ksp, &data, &size) )
        return array_r(size, data)

    def setMonitor(self, monitor, *args, **kargs):
        if monitor is None: return
        KSP_setMon(self.ksp, (monitor, args, kargs))

    def getMonitor(self):
        return KSP_getMon(self.ksp)

    def cancelMonitor(self):
        KSP_clsMon(self.ksp)

    # --- xxx ---

    def setUp(self):
        CHKERR( KSPSetUp(self.ksp) )

    def setUpOnBlocks(self):
        CHKERR( KSPSetUpOnBlocks(self.ksp) )

    def solve(self, Vec b not None, Vec x not None):
        CHKERR( KSPSolve(self.ksp, b.vec, x.vec) )

    def solveTranspose(self, Vec b not None, Vec x not None):
        CHKERR( KSPSolveTranspose(self.ksp, b.vec, x.vec) )

    def getIterationNumber(self):
        cdef PetscInt ival = 0
        CHKERR( KSPGetIterationNumber(self.ksp, &ival) )
        return ival

    def getResidualNorm(self):
        cdef PetscReal rval = 0
        CHKERR(KSPGetResidualNorm(self.ksp, &rval) )
        return rval

    def getConvergedReason(self):
        cdef PetscKSPConvergedReason reason
        reason = KSP_CONVERGED_ITERATING
        CHKERR( KSPGetConvergedReason(self.ksp, &reason) )
        return reason

    def getRhs(self):
        cdef Vec vec = Vec()
        CHKERR( KSPGetRhs(self.ksp, &vec.vec) )
        PetscIncref(<PetscObject>vec.vec)
        return vec

    def getSolution(self):
        cdef Vec vec = Vec()
        CHKERR( KSPGetSolution(self.ksp, &vec.vec) )
        PetscIncref(<PetscObject>vec.vec)
        return vec

    # --- xxx ---

    def createPython(self, context=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_WORLD)
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

    property vec_rhs:
        def __get__(self):
            return self.getRhs()

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

    # --- convergence ---

    property reason:
        def __get__(self):
            return self.getConvergedReason()

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
