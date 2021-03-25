# --------------------------------------------------------------------

class TAOType:
    """
    TAO Solver Types
    """
    LMVM     = S_(TAOLMVM)
    NLS      = S_(TAONLS)
    NTR      = S_(TAONTR)
    NTL      = S_(TAONTL)
    CG       = S_(TAOCG)
    TRON     = S_(TAOTRON)
    OWLQN    = S_(TAOOWLQN)
    BMRM     = S_(TAOBMRM)
    BLMVM    = S_(TAOBLMVM)
    BQNLS    = S_(TAOBQNLS)
    BNCG     = S_(TAOBNCG)
    BNLS     = S_(TAOBNLS)
    BNTR     = S_(TAOBNTR)
    BNTL     = S_(TAOBNTL)
    BQNKLS   = S_(TAOBQNKLS)
    BQNKTR   = S_(TAOBQNKTR)
    BQNKTL   = S_(TAOBQNKTL)
    BQPIP    = S_(TAOBQPIP)
    GPCG     = S_(TAOGPCG)
    NM       = S_(TAONM)
    POUNDERS = S_(TAOPOUNDERS)
    BRGN     = S_(TAOBRGN)
    LCL      = S_(TAOLCL)
    SSILS    = S_(TAOSSILS)
    SSFLS    = S_(TAOSSFLS)
    ASILS    = S_(TAOASILS)
    ASFLS    = S_(TAOASFLS)
    IPM      = S_(TAOIPM)
    PDIPM    = S_(TAOPDIPM)
    SHELL    = S_(TAOSHELL)
    ADMM     = S_(TAOADMM)
    ALMM     = S_(TAOALMM)

class TAOConvergedReason:
    """
    TAO Solver Termination Reasons
    """
    # iterating
    CONTINUE_ITERATING    = TAO_CONTINUE_ITERATING    # iterating
    CONVERGED_ITERATING   = TAO_CONTINUE_ITERATING    # iterating
    ITERATING             = TAO_CONTINUE_ITERATING    # iterating
    # converged
    CONVERGED_GATOL       = TAO_CONVERGED_GATOL       # ||g(X)|| < gatol
    CONVERGED_GRTOL       = TAO_CONVERGED_GRTOL       # ||g(X)||/f(X)  < grtol
    CONVERGED_GTTOL       = TAO_CONVERGED_GTTOL       # ||g(X)||/||g(X0)|| < gttol
    CONVERGED_STEPTOL     = TAO_CONVERGED_STEPTOL     # small step size
    CONVERGED_MINF        = TAO_CONVERGED_MINF        # f(X) < F_min
    CONVERGED_USER        = TAO_CONVERGED_USER        # user defined
    # diverged
    DIVERGED_MAXITS       = TAO_DIVERGED_MAXITS       #
    DIVERGED_NAN          = TAO_DIVERGED_NAN          #
    DIVERGED_MAXFCN       = TAO_DIVERGED_MAXFCN       #
    DIVERGED_LS_FAILURE   = TAO_DIVERGED_LS_FAILURE   #
    DIVERGED_TR_REDUCTION = TAO_DIVERGED_TR_REDUCTION #
    DIVERGED_USER         = TAO_DIVERGED_USER         # user defined

# --------------------------------------------------------------------

cdef class TAO(Object):

    """
    TAO Solver
    """

    Type   = TAOType
    Reason = TAOConvergedReason

    def __cinit__(self):
        self.obj = <PetscObject*> &self.tao
        self.tao = NULL

    def view(self, Viewer viewer=None):
        """
        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( TaoView(self.tao, vwr) )

    def destroy(self):
        """
        """
        CHKERR( TaoDestroy(&self.tao) )
        return self

    def create(self, comm=None):
        """
        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscTAO newtao = NULL
        CHKERR( TaoCreate(ccomm, &newtao) )
        PetscCLEAR(self.obj); self.tao = newtao
        return self

    def setType(self, tao_type):
        """
        """
        cdef PetscTAOType ctype = NULL
        tao_type = str2bytes(tao_type, &ctype)
        CHKERR( TaoSetType(self.tao, ctype) )

    def getType(self):
        """
        """
        cdef PetscTAOType ctype = NULL
        CHKERR( TaoGetType(self.tao, &ctype) )
        return bytes2str(ctype)

    def setOptionsPrefix(self, prefix):
        """
        """
        cdef const char *cprefix = NULL
        prefix = str2bytes(prefix, &cprefix)
        CHKERR( TaoSetOptionsPrefix(self.tao, cprefix) )

    def getOptionsPrefix(self):
        """
        """
        cdef const char *prefix = NULL
        CHKERR( TaoGetOptionsPrefix(self.tao, &prefix) )
        return bytes2str(prefix)

    def setFromOptions(self):
        """
        """
        CHKERR( TaoSetFromOptions(self.tao) )

    def setUp(self):
        """
        """
        CHKERR( TaoSetUp(self.tao) )

    #

    def setInitialTrustRegionRadius(self, radius):
        cdef PetscReal cradius = asReal(radius)
        CHKERR( TaoSetInitialTrustRegionRadius(self.tao, cradius) )

    # --------------

    def setAppCtx(self, appctx):
        self.set_attr("__appctx__", appctx)

    def getAppCtx(self):
        return self.get_attr("__appctx__")

    def setInitial(self, Vec x):
        """
        """
        CHKERR( TaoSetInitialVector(self.tao, x.vec) )

    def setObjective(self, objective, args=None, kargs=None):
        """
        """
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (objective, args, kargs)
        self.set_attr("__objective__", context)
        CHKERR( TaoSetObjectiveRoutine(self.tao, TAO_Objective, <void*>context) )

    def setResidual(self, residual, Vec R=None, args=None, kargs=None):
        """
        """
        cdef PetscVec Rvec = NULL
        if R is not None: Rvec = R.vec
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (residual, args, kargs)
        self.set_attr("__residual__", context)
        CHKERR( TaoSetResidualRoutine(self.tao, Rvec, TAO_Residual, <void*>context) )

    def setJacobianResidual(self, jacobian, Mat J=None, Mat P=None, args=None, kargs=None):
        """
        """
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (jacobian, args, kargs)
        self.set_attr("__jacobian_residual__", context)
        CHKERR( TaoSetJacobianResidualRoutine(self.tao, Jmat, Pmat, TAO_JacobianResidual, <void*>context) )

    def setGradient(self, gradient, args=None, kargs=None):
        """
        """
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (gradient, args, kargs)
        self.set_attr("__gradient__", context)
        CHKERR( TaoSetGradientRoutine(self.tao, TAO_Gradient, <void*>context) )

    def setObjectiveGradient(self, objgrad, args=None, kargs=None):
        """
        """
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (objgrad, args, kargs)
        self.set_attr("__objgrad__", context)
        CHKERR( TaoSetObjectiveAndGradientRoutine(self.tao, TAO_ObjGrad, <void*>context) )

    def setVariableBounds(self, varbounds, args=None, kargs=None):
        """
        """
        cdef Vec xl=None, xu=None
        if (isinstance(varbounds, list) or isinstance(varbounds, tuple)):
            ol, ou = varbounds
            xl = <Vec?> ol; xu = <Vec?> ou
            CHKERR( TaoSetVariableBounds(self.tao, xl.vec, xu.vec) )
            return
        if isinstance(varbounds, Vec):
            ol = varbounds; ou = args
            xl = <Vec?> ol; xu = <Vec?> ou
            CHKERR( TaoSetVariableBounds(self.tao, xl.vec, xu.vec) )
            return
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (varbounds, args, kargs)
        self.set_attr("__varbounds__", context)
        CHKERR( TaoSetVariableBoundsRoutine(self.tao, TAO_VarBounds, <void*>context) )

    def setConstraints(self, constraints, Vec C=None, args=None, kargs=None):
        """
        """
        cdef PetscVec Cvec=NULL
        if C is not None: Cvec = C.vec
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (constraints, args, kargs)
        self.set_attr("__constraints__", context)
        CHKERR( TaoSetConstraintsRoutine(self.tao, Cvec, TAO_Constraints, <void*>context) )

    def setHessian(self, hessian, Mat H=None, Mat P=None,
                   args=None, kargs=None):
        cdef PetscMat Hmat=NULL
        if H is not None: Hmat = H.mat
        cdef PetscMat Pmat = Hmat
        if P is not None: Pmat = P.mat
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (hessian, args, kargs)
        self.set_attr("__hessian__", context)
        CHKERR( TaoSetHessianRoutine(self.tao, Hmat, Pmat, TAO_Hessian, <void*>context) )

    def setJacobian(self, jacobian, Mat J=None, Mat P=None,
                    args=None, kargs=None):
        """
        """
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (jacobian, args, kargs)
        self.set_attr("__jacobian__", context)
        CHKERR( TaoSetJacobianRoutine(self.tao, Jmat, Pmat, TAO_Jacobian, <void*>context) )

    #

    def setStateDesignIS(self, IS state=None, IS design=None):
        """
        """
        cdef PetscIS s_is = NULL, d_is = NULL
        if state  is not None: s_is = state.iset
        if design is not None: d_is = design.iset
        CHKERR( TaoSetStateDesignIS(self.tao, s_is, d_is) )

    def setJacobianState(self, jacobian_state, Mat J=None, Mat P=None, Mat I=None,
                         args=None, kargs=None):
        """
        """
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        cdef PetscMat Imat = NULL
        if I is not None: Imat = I.mat
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (jacobian_state, args, kargs)
        self.set_attr("__jacobian_state__", context)
        CHKERR( TaoSetJacobianStateRoutine(self.tao, Jmat, Pmat, Imat,
                                           TAO_JacobianState, <void*>context) )

    def setJacobianDesign(self, jacobian_design, Mat J=None,
                          args=None, kargs=None):
        """
        """
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (jacobian_design, args, kargs)
        self.set_attr("__jacobian_design__", context)
        CHKERR( TaoSetJacobianDesignRoutine(self.tao, Jmat,
                                            TAO_JacobianDesign, <void*>context) )

    # --------------

    def computeObjective(self, Vec x):
        """
        """
        cdef PetscReal f = 0
        CHKERR( TaoComputeObjective(self.tao, x.vec, &f) )
        return toReal(f)

    def computeResidual(self, Vec x, Vec f):
        """
        """
        CHKERR( TaoComputeResidual(self.tao, x.vec, f.vec) )

    def computeGradient(self, Vec x, Vec g):
        """
        """
        CHKERR( TaoComputeGradient(self.tao, x.vec, g.vec) )

    def computeObjectiveGradient(self, Vec x, Vec g):
        """
        """
        cdef PetscReal f = 0
        CHKERR( TaoComputeObjectiveAndGradient(self.tao, x.vec, &f, g.vec) )
        return toReal(f)

    def computeDualVariables(self, Vec xl, Vec xu):
        """
        """
        CHKERR( TaoComputeDualVariables(self.tao, xl.vec, xu.vec) )

    def computeVariableBounds(self, Vec xl, Vec xu):
        """
        """
        CHKERR( TaoComputeVariableBounds(self.tao) )
        cdef PetscVec Lvec = NULL, Uvec = NULL
        CHKERR( TaoGetVariableBounds(self.tao, &Lvec, &Uvec) )
        if xl.vec != NULL:
            if Lvec != NULL:
                CHKERR( VecCopy(Lvec, xl.vec) )
            else:
                CHKERR( VecSet(xl.vec, <PetscScalar>PETSC_NINFINITY) )
        if xu.vec != NULL:
            if Uvec != NULL:
                CHKERR( VecCopy(Uvec, xu.vec) )
            else:
                CHKERR( VecSet(xu.vec, <PetscScalar>PETSC_INFINITY) )

    def computeConstraints(self, Vec x, Vec c):
        """
        """
        CHKERR( TaoComputeConstraints(self.tao, x.vec, c.vec) )

    def computeHessian(self, Vec x, Mat H, Mat P=None):
        """
        """
        cdef PetscMat hmat = H.mat, pmat = H.mat
        if P is not None: pmat = P.mat
        CHKERR( TaoComputeHessian(self.tao, x.vec, hmat, pmat) )

    def computeJacobian(self, Vec x, Mat J, Mat P=None):
        """
        """
        cdef PetscMat jmat = J.mat, pmat = J.mat
        if P is not None: pmat = P.mat
        CHKERR( TaoComputeJacobian(self.tao, x.vec, jmat, pmat) )

    # --------------

    #

    def setTolerances(self, gatol=None, grtol=None, gttol=None):
        """
        """
        cdef PetscReal _gatol=PETSC_DEFAULT, _grtol=PETSC_DEFAULT, _gttol=PETSC_DEFAULT
        if gatol is not None: _gatol = asReal(gatol)
        if grtol is not None: _grtol = asReal(grtol)
        if gttol is not None: _gttol = asReal(gttol)
        CHKERR( TaoSetTolerances(self.tao, _gatol, _grtol, _gttol) )

    def getTolerances(self):
        """
        """
        cdef PetscReal _gatol=PETSC_DEFAULT, _grtol=PETSC_DEFAULT, _gttol=PETSC_DEFAULT
        CHKERR( TaoGetTolerances(self.tao, &_gatol, &_grtol, &_gttol) )
        return (toReal(_gatol), toReal(_grtol), toReal(_gttol))

    def setConstraintTolerances(self, catol=None, crtol=None):
        """
        """
        cdef PetscReal _catol=PETSC_DEFAULT, _crtol=PETSC_DEFAULT
        if catol is not None: _catol = asReal(catol)
        if crtol is not None: _crtol = asReal(crtol)
        CHKERR( TaoSetConstraintTolerances(self.tao, _catol, _crtol) )

    def getConstraintTolerances(self):
        """
        """
        cdef PetscReal _catol=PETSC_DEFAULT, _crtol=PETSC_DEFAULT
        CHKERR( TaoGetConstraintTolerances(self.tao, &_catol, &_crtol) )
        return (toReal(_catol), toReal(_crtol))

    def setConvergenceTest(self, converged, args=None, kargs=None):
        """
        """
        if converged is None:
            CHKERR( TaoSetConvergenceTest(self.tao, TaoDefaultConvergenceTest, NULL) )
            self.set_attr('__converged__', None)
        else:
            if args is None: args = ()
            if kargs is None: kargs = {}
            self.set_attr('__converged__', (converged, args, kargs))
            CHKERR( TaoSetConvergenceTest(self.tao, TAO_Converged, NULL) )

    def getConvergenceTest(self):
        """
        """
        return self.get_attr('__converged__')

    def setConvergedReason(self, reason):
        """
        """
        cdef PetscTAOConvergedReason creason = reason
        CHKERR( TaoSetConvergedReason(self.tao, creason) )

    def getConvergedReason(self):
        """
        """
        cdef PetscTAOConvergedReason creason = TAO_CONTINUE_ITERATING
        CHKERR( TaoGetConvergedReason(self.tao, &creason) )
        return creason

    def setMonitor(self, monitor, args=None, kargs=None):
        """
        """
        if monitor is None: return
        cdef object monitorlist = self.get_attr('__monitor__')
        if monitorlist is None:
            CHKERR( TaoSetMonitor(self.tao, TAO_Monitor, NULL, NULL) )
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            self.set_attr('__monitor__',  [(monitor, args, kargs)])
        else:
            monitorlist.append((monitor, args, kargs))

    def getMonitor(self):
        """
        """
        return self.get_attr('__monitor__')

    def cancelMonitor(self):
        """
        """
        CHKERR( TaoCancelMonitors(self.tao) )
        self.set_attr('__monitor__',  None)

    #

    def solve(self, Vec x=None):
        """
        """
        if x is not None:
            CHKERR( TaoSetInitialVector(self.tao, x.vec) )
        CHKERR( TaoSolve(self.tao) )

    def getSolution(self):
        """
        """
        cdef Vec vec = Vec()
        CHKERR( TaoGetSolutionVector(self.tao, &vec.vec) )
        PetscINCREF(vec.obj)
        return vec

    def getGradient(self):
        """
        """
        cdef Vec vec = Vec()
        CHKERR( TaoGetGradientVector(self.tao, &vec.vec) )
        PetscINCREF(vec.obj)
        return vec

    def setGradientNorm(self, Mat mat):
        """
        """
        CHKERR( TaoSetGradientNorm(self.tao, mat.mat) )

    def getGradientNorm(self):
        """
        """
        cdef Mat mat = Mat()
        CHKERR( TaoGetGradientNorm(self.tao, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

    def setLMVMH0(self, Mat mat):
        """
        """
        CHKERR( TaoLMVMSetH0(self.tao, mat.mat) )

    def getLMVMH0(self):
        """
        """
        cdef Mat mat = Mat()
        CHKERR( TaoLMVMGetH0(self.tao, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

    def getLMVMH0KSP(self):
        """
        """
        cdef KSP ksp = KSP()
        CHKERR( TaoLMVMGetH0KSP(self.tao, &ksp.ksp) )
        PetscINCREF(ksp.obj)
        return ksp

    def getVariableBounds(self):
        """
        """
        cdef Vec xl = Vec(), xu = Vec()
        CHKERR( TaoGetVariableBounds(self.tao, &xl.vec, &xu.vec) )
        PetscINCREF(xl.obj); PetscINCREF(xu.obj)
        return (xl, xu)

    def getIterationNumber(self):
        """
        """
        cdef PetscInt its=0
        CHKERR( TaoGetSolutionStatus(self.tao, &its, NULL, NULL, NULL, NULL, NULL) )
        return toInt(its)

    def getObjectiveValue(self):
        """
        """
        cdef PetscReal fval=0
        CHKERR( TaoGetSolutionStatus(self.tao, NULL, &fval, NULL, NULL, NULL, NULL) )
        return toReal(fval)

    getFunctionValue = getObjectiveValue

    def getConvergedReason(self):
        """
        """
        cdef PetscTAOConvergedReason reason = TAO_CONTINUE_ITERATING
        CHKERR( TaoGetConvergedReason(self.tao, &reason) )
        return reason

    def getSolutionNorm(self):
        """
        """
        cdef PetscReal gnorm=0
        cdef PetscReal cnorm=0
        cdef PetscReal fval=0
        CHKERR( TaoGetSolutionStatus(self.tao, NULL, &fval, &gnorm, &cnorm, NULL, NULL) )
        return (toReal(fval), toReal(gnorm), toReal(cnorm))

    def getSolutionStatus(self):
        """
        """
        cdef PetscInt its=0
        cdef PetscReal fval=0, gnorm=0, cnorm=0, xdiff=0
        cdef PetscTAOConvergedReason reason = TAO_CONTINUE_ITERATING
        CHKERR( TaoGetSolutionStatus(self.tao, &its,
                                     &fval, &gnorm, &cnorm, &xdiff,
                                     &reason) )
        return (toInt(its), toReal(fval),
                toReal(gnorm), toReal(cnorm),
                toReal(xdiff), reason)

    def getKSP(self):
        """
        """
        cdef KSP ksp = KSP()
        CHKERR( TaoGetKSP(self.tao, &ksp.ksp) )
        PetscINCREF(ksp.obj)
        return ksp

    # BRGN routines

    def getBRGNSubsolver(self):
        """
        """
        cdef TAO subsolver = TAO()
        CHKERR( TaoBRGNGetSubsolver(self.tao, &subsolver.tao) )
        PetscINCREF(subsolver.obj)
        return subsolver

    def setBRGNRegularizerObjectiveGradient(self, objgrad, args=None, kargs=None):
        """
        """
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (objgrad, args, kargs)
        self.set_attr("__brgnregobjgrad__", context)
        CHKERR( TaoBRGNSetRegularizerObjectiveAndGradientRoutine(self.tao, TAO_BRGNRegObjGrad, <void*>context) )

    def setBRGNRegularizerHessian(self, hessian, Mat H=None, args=None, kargs=None):
        cdef PetscMat Hmat=NULL
        if H is not None: Hmat = H.mat
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (hessian, args, kargs)
        self.set_attr("__brgnreghessian__", context)
        CHKERR( TaoBRGNSetRegularizerHessianRoutine(self.tao, Hmat, TAO_BRGNRegHessian, <void*>context) )

    def setBRGNRegularizerWeight(self, weight):
        """
        """
        cdef PetscReal cweight = asReal(weight)
        CHKERR( TaoBRGNSetRegularizerWeight(self.tao, cweight) )

    def setBRGNSmoothL1Epsilon(self, epsilon):
        """
        """
        cdef PetscReal ceps = asReal(epsilon)
        CHKERR( TaoBRGNSetL1SmoothEpsilon(self.tao, ceps) )

    def setBRGNDictionaryMatrix(self, Mat D):
        """
        """
        CHKERR( TaoBRGNSetDictionaryMatrix(self.tao, D.mat) )

    def getBRGNDampingVector(self):
        """
        """
        cdef Vec damp = Vec()
        CHKERR( TaoBRGNGetDampingVector(self.tao, &damp.vec) )
        PetscINCREF(damp.obj)
        return damp

    # --- application context ---

    property appctx:
        def __get__(self):
            return self.getAppCtx()
        def __set__(self, value):
            self.setAppCtx(value)

    # --- linear solver ---

    property ksp:
        def __get__(self):
            return self.getKSP()

    # --- tolerances ---

    property ftol:
        def __get__(self):
            return self.getFunctionTolerances()
        def __set__(self, value):
            if isinstance(value, (tuple, list)):
                self.setFunctionTolerances(*value)
            elif isinstance(value, dict):
                self.setFunctionTolerances(**value)
            else:
                raise TypeError("expecting tuple/list or dict")

    property gtol:
        def __get__(self):
            return self.getGradientTolerances()
        def __set__(self, value):
            if isinstance(value, (tuple, list)):
                self.getGradientTolerances(*value)
            elif isinstance(value, dict):
                self.getGradientTolerances(**value)
            else:
                raise TypeError("expecting tuple/list or dict")

    property ctol:
        def __get__(self):
            return self.getConstraintTolerances()
        def __set__(self, value):
            if isinstance(value, (tuple, list)):
                self.getConstraintTolerances(*value)
            elif isinstance(value, dict):
                self.getConstraintTolerances(**value)
            else:
                raise TypeError("expecting tuple/list or dict")

    # --- iteration ---

    property its:
        def __get__(self):
            return self.getIterationNumber()

    property gnorm:
        def __get__(self):
            return self.getSolutionNorm()[1]

    property cnorm:
        def __get__(self):
            return self.getSolutionNorm()[2]

    property solution:
        def __get__(self):
            return self.getSolution()

    property objective:
        def __get__(self):
            return self.getObjectiveValue()

    property function:
        def __get__(self):
            return self.getFunctionValue()

    property gradient:
        def __get__(self):
            return self.getGradient()

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

del TAOType
del TAOConvergedReason

# --------------------------------------------------------------------
