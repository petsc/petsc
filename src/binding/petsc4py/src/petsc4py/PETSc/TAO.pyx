# --------------------------------------------------------------------

class TAOType:
    """TAO solver type."""
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
    PYTHON   = S_(TAOPYTHON)

class TAOConvergedReason:
    """TAO solver termination reason."""
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
    """Optimization solver.

    TAO is described in the `PETSc manual <petsc:manual/tao>`.

    See Also
    --------
    petsc.Tao

    """

    Type = TAOType
    ConvergedReason = TAOConvergedReason
    # FIXME backward compatibility
    Reason = TAOConvergedReason

    def __cinit__(self):
        self.obj = <PetscObject*> &self.tao
        self.tao = NULL

    def view(self, Viewer viewer=None) -> None:
        """View the solver.

        Collective.

        Parameters
        ----------
        viewer
          A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        petsc.TaoView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( TaoView(self.tao, vwr) )

    def destroy(self) -> Self:
        """Destroy the solver.

        Collective.

        See Also
        --------
        petsc.TaoDestroy

        """
        CHKERR( TaoDestroy(&self.tao) )
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create a TAO solver.

        Collective.

        Parameters
        ----------
        comm
          The communicator associated with the object. Defaults to `Sys.getDefaultComm`.

        See Also
        --------
        Sys.getDefaultComm, petsc.TaoCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscTAO newtao = NULL
        CHKERR( TaoCreate(ccomm, &newtao) )
        PetscCLEAR(self.obj); self.tao = newtao
        return self

    def setType(self, tao_type: TAO.Type | str) -> None:
        """Set the type of the TAO solver.

        Logically collective.

        Parameters
        ----------
        tao_type
          The type of the solver.

        See Also
        --------
        getType, petsc.TaoSetType

        """
        cdef PetscTAOType ctype = NULL
        tao_type = str2bytes(tao_type, &ctype)
        CHKERR( TaoSetType(self.tao, ctype) )

    def getType(self) -> str:
        """Return the type of the TAO solver object.

        Not collective.

        See Also
        --------
        setType, petsc.TaoGetType

        """
        cdef PetscTAOType ctype = NULL
        CHKERR( TaoGetType(self.tao, &ctype) )
        return bytes2str(ctype)

    def setOptionsPrefix(self, prefix: str) -> None:
        """Set the prefix used for searching for options in the database.

        Logically collective.

        See Also
        --------
        petsc_options, petsc.TaoSetOptionsPrefix

        """
        cdef const char *cprefix = NULL
        prefix = str2bytes(prefix, &cprefix)
        CHKERR( TaoSetOptionsPrefix(self.tao, cprefix) )

    def appendOptionsPrefix(self, prefix: str) -> None:
        """Append to the prefix used for searching for options in the database.

        Logically collective.

        See Also
        --------
        petsc_options, setOptionsPrefix, petsc.TaoAppendOptionsPrefix

        """
        cdef const char *cprefix = NULL
        prefix = str2bytes(prefix, &cprefix)
        CHKERR( TaoAppendOptionsPrefix(self.tao, cprefix) )

    def getOptionsPrefix(self) -> str:
        """Return the prefix used for searching for options in the database.

        Not collective.

        See Also
        --------
        petsc_options, setOptionsPrefix, petsc.TaoGetOptionsPrefix

        """
        cdef const char *prefix = NULL
        CHKERR( TaoGetOptionsPrefix(self.tao, &prefix) )
        return bytes2str(prefix)

    def setFromOptions(self) -> None:
        """Configure the TAO solver from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.TaoSetFromOptions

        """
        CHKERR( TaoSetFromOptions(self.tao) )

    def setUp(self) -> None:
        """Set up the internal data structures for using the solver.

        Collective.

        See Also
        --------
        petsc.TaoSetUp

        """
        CHKERR( TaoSetUp(self.tao) )

    #

    def setInitialTrustRegionRadius(self, radius: float) -> None:
        """Set the initial trust region radius.

        Collective.

        See Also
        --------
        petsc.TaoSetInitialTrustRegionRadius

        """
        cdef PetscReal cradius = asReal(radius)
        CHKERR( TaoSetInitialTrustRegionRadius(self.tao, cradius) )

    # --------------

    def setAppCtx(self, appctx: Any) -> None:
        """Set the application context."""
        self.set_attr("__appctx__", appctx)

    def getAppCtx(self) -> Any:
        """Return the application context."""
        return self.get_attr("__appctx__")

    def setSolution(self, Vec x) -> None:
        """Set the vector used to store the solution.

        Collective.

        See Also
        --------
        getSolution, petsc.TaoSetSolution

        """
        CHKERR( TaoSetSolution(self.tao, x.vec) )

    def setObjective(self, objective: TAOObjectiveFunction, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the objective function evaluation callback.

        Logically collective.

        Parameters
        ----------
        objective
          The objective function callback.
        args
          Positional arguments for the callback.
        kargs
          Keyword arguments for the callback.

        See Also
        --------
        setGradient, setObjectiveGradient, petsc.TaoSetObjective

        """
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (objective, args, kargs)
        self.set_attr("__objective__", context)
        CHKERR( TaoSetObjective(self.tao, TAO_Objective, <void*>context) )

    def setResidual(self, residual: TAOResidualFunction, Vec R=None, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the residual evaluation callback for least-squares applications.

        Logically collective.

        Parameters
        ----------
        residual
          The residual callback.
        R
          The vector to store the residual.
        args
          Positional arguments for the callback.
        kargs
          Keyword arguments for the callback.

        See Also
        --------
        setJacobianResidual, petsc.TaoSetResidualRoutine

        """
        cdef PetscVec Rvec = NULL
        if R is not None: Rvec = R.vec
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (residual, args, kargs)
        self.set_attr("__residual__", context)
        CHKERR( TaoSetResidualRoutine(self.tao, Rvec, TAO_Residual, <void*>context) )

    def setJacobianResidual(self, jacobian: TAOJacobianResidualFunction, Mat J=None, Mat P=None, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute the least-squares residual Jacobian.

        Logically collective.

        Parameters
        ----------
        jacobian
          The Jacobian callback.
        J
          The matrix to store the Jacobian.
        P
          The matrix to construct the preconditioner.
        args
          Positional arguments for the callback.
        kargs
          Keyword arguments for the callback.

        See Also
        --------
        setResidual, petsc.TaoSetJacobianResidualRoutine

        """
        cdef PetscMat Jmat = NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (jacobian, args, kargs)
        self.set_attr("__jacobian_residual__", context)
        CHKERR( TaoSetJacobianResidualRoutine(self.tao, Jmat, Pmat, TAO_JacobianResidual, <void*>context) )

    def setGradient(self, gradient: TAOGradientFunction, Vec g=None, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the gradient evaluation callback.

        Logically collective.

        Parameters
        ----------
        gradient
          The gradient callback.
        g
          The vector to store the gradient.
        args
          Positional arguments for the callback.
        kargs
          Keyword arguments for the callback.

        See Also
        --------
        setObjective, setObjectiveGradient, setHessian, petsc.TaoSetGradient

        """
        cdef PetscVec gvec = NULL
        if g is not None: gvec = g.vec
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (gradient, args, kargs)
        self.set_attr("__gradient__", context)
        CHKERR( TaoSetGradient(self.tao, gvec, TAO_Gradient, <void*>context) )

    def getGradient(self) -> tuple[Vec, TAOGradientFunction]:
        """Return the vector used to store the gradient and the evaluation callback.

        Not collective.

        See Also
        --------
        setGradient, setHessian, petsc.TaoGetGradient

        """
        cdef Vec vec = Vec()
        CHKERR( TaoGetGradient(self.tao, &vec.vec, NULL, NULL) )
        PetscINCREF(vec.obj)
        cdef object gradient = self.get_attr("__gradient__")
        return (vec, gradient)

    def setObjectiveGradient(self, objgrad: TAOObjectiveGradientFunction, Vec g=None, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the objective function and gradient evaluation callback.

        Logically collective.

        Parameters
        ----------
        objgrad
          The objective function and gradient callback.
        g
          The vector to store the gradient.
        args
          Positional arguments for the callback.
        kargs
          Keyword arguments for the callback.

        See Also
        --------
        setObjective, setGradient, setHessian, getObjectiveAndGradient, petsc.TaoSetObjectiveAndGradient

        """
        cdef PetscVec gvec = NULL
        if g is not None: gvec = g.vec
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (objgrad, args, kargs)
        self.set_attr("__objgrad__", context)
        CHKERR( TaoSetObjectiveAndGradient(self.tao, gvec, TAO_ObjGrad, <void*>context) )

    def getObjectiveAndGradient(self) -> tuple[Vec, TAOObjectiveGradientFunction]:
        """Return the vector used to store the gradient and the evaluation callback.

        Not collective.

        See Also
        --------
        setObjectiveGradient, petsc.TaoGetObjectiveAndGradient

        """
        cdef Vec vec = Vec()
        CHKERR( TaoGetObjectiveAndGradient(self.tao, &vec.vec, NULL, NULL) )
        PetscINCREF(vec.obj)
        cdef object objgrad = self.get_attr("__objgrad__")
        return (vec, objgrad)

    def setVariableBounds(self, varbounds: tuple[Vec, Vec] | TAOVariableBoundsFunction, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the upper and lower bounds for the optimization problem.

        Logically collective.

        Parameters
        ----------
        varbounds
          Either a tuple of `Vec` or a `TAOVariableBoundsFunction` callback.
        args
          Positional arguments for the callback.
        kargs
          Keyword arguments for the callback.

        See Also
        --------
        petsc.TaoSetVariableBounds, petsc.TaoSetVariableBoundsRoutine

        """
        cdef Vec xl = None, xu = None
        if (isinstance(varbounds, list) or isinstance(varbounds, tuple)):
            ol, ou = varbounds
            xl = <Vec?> ol; xu = <Vec?> ou
            CHKERR( TaoSetVariableBounds(self.tao, xl.vec, xu.vec) )
            return
        if isinstance(varbounds, Vec): #FIXME
            ol = varbounds; ou = args
            xl = <Vec?> ol; xu = <Vec?> ou
            CHKERR( TaoSetVariableBounds(self.tao, xl.vec, xu.vec) )
            return
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (varbounds, args, kargs)
        self.set_attr("__varbounds__", context)
        CHKERR( TaoSetVariableBoundsRoutine(self.tao, TAO_VarBounds, <void*>context) )

    def setConstraints(self, constraints: TAOConstraintsFunction, Vec C=None, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute constraints.

        Logically collective.

        Parameters
        ----------
        constraints
          The callback.
        C
          The vector to hold the constraints.
        args
          Positional arguments for the callback.
        kargs
          Keyword arguments for the callback.

        See Also
        --------
        petsc.TaoSetConstraintsRoutine

        """
        cdef PetscVec Cvec = NULL
        if C is not None: Cvec = C.vec
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (constraints, args, kargs)
        self.set_attr("__constraints__", context)
        CHKERR( TaoSetConstraintsRoutine(self.tao, Cvec, TAO_Constraints, <void*>context) )

    def setHessian(self, hessian: TAOHessianFunction, Mat H=None, Mat P=None,
                   args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute the Hessian matrix.

        Logically collective.

        Parameters
        ----------
        hessian
          The Hessian callback.
        H
          The matrix to store the Hessian.
        P
          The matrix to construct the preconditioner.
        args
          Positional arguments for the callback.
        kargs
          Keyword arguments for the callback.

        See Also
        --------
        getHessian, setObjective, setObjectiveGradient, setGradient, petsc.TaoSetHessian

        """
        cdef PetscMat Hmat = NULL
        if H is not None: Hmat = H.mat
        cdef PetscMat Pmat = Hmat
        if P is not None: Pmat = P.mat
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (hessian, args, kargs)
        self.set_attr("__hessian__", context)
        CHKERR( TaoSetHessian(self.tao, Hmat, Pmat, TAO_Hessian, <void*>context) )

    def getHessian(self) -> tuple[Mat, Mat, TAOHessianFunction]:
        """Return the matrices used to store the Hessian and the evaluation callback.

        Not collective.

        See Also
        --------
        setHessian, petsc.TaoGetHessian

        """
        cdef Mat J = Mat()
        cdef Mat P = Mat()
        CHKERR( TaoGetHessian(self.tao, &J.mat, &P.mat, NULL, NULL) )
        PetscINCREF(J.obj)
        PetscINCREF(P.obj)
        cdef object hessian = self.get_attr("__hessian__")
        return (J, P, hessian)

    def setJacobian(self, jacobian: TAOJacobianFunction, Mat J=None, Mat P=None,
                    args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute the Jacobian.

        Logically collective.

        Parameters
        ----------
        jacobian
          The Jacobian callback.
        J
          The matrix to store the Jacobian.
        P
          The matrix to construct the preconditioner.
        args
          Positional arguments for the callback.
        kargs
          Keyword arguments for the callback.

        See Also
        --------
        petsc.TaoSetJacobianRoutine

        """
        cdef PetscMat Jmat = NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (jacobian, args, kargs)
        self.set_attr("__jacobian__", context)
        CHKERR( TaoSetJacobianRoutine(self.tao, Jmat, Pmat, TAO_Jacobian, <void*>context) )

    #

    def setStateDesignIS(self, IS state=None, IS design=None) -> None:
        """Set the index sets indicating state and design variables.

        Collective.

        See Also
        --------
        petsc.TaoSetStateDesignIS

        """
        cdef PetscIS s_is = NULL, d_is = NULL
        if state  is not None: s_is = state.iset
        if design is not None: d_is = design.iset
        CHKERR( TaoSetStateDesignIS(self.tao, s_is, d_is) )

    def setJacobianState(self, jacobian_state, Mat J=None, Mat P=None, Mat I=None,
                         args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set Jacobian state callback.

        Logically collective.

        See Also
        --------
        petsc.TaoSetJacobianStateRoutine

        """
        cdef PetscMat Jmat = NULL
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
                          args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set Jacobian design callback.

        Logically collective.

        See Also
        --------
        petsc.TaoSetJacobianDesignRoutine

        """
        cdef PetscMat Jmat = NULL
        if J is not None: Jmat = J.mat
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (jacobian_design, args, kargs)
        self.set_attr("__jacobian_design__", context)
        CHKERR( TaoSetJacobianDesignRoutine(self.tao, Jmat,
                                            TAO_JacobianDesign, <void*>context) )


    def setEqualityConstraints(self, equality_constraints, Vec c,
                               args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set equality constraints callback.

        Logically collective.

        See Also
        --------
        petsc.TaoSetEqualityConstraintsRoutine

        """
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (equality_constraints, args, kargs)
        self.set_attr("__equality_constraints__", context)
        CHKERR( TaoSetEqualityConstraintsRoutine(self.tao, c.vec,
                                                 TAO_EqualityConstraints, <void*>context) )


    def setJacobianEquality(self, jacobian_equality, Mat J=None, Mat P=None,
                            args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set Jacobian equality constraints callback.

        Logically collective.

        See Also
        --------
        petsc.TaoSetJacobianEqualityRoutine

        """
        cdef PetscMat Jmat = NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (jacobian_equality, args, kargs)
        self.set_attr("__jacobian_equality__", context)
        CHKERR( TaoSetJacobianEqualityRoutine(self.tao, Jmat, Pmat,
                                              TAO_JacobianEquality, <void*>context) )

    def setUpdate(self, update: TAOUpdateFunction, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute update at each optimization step.

        Logically collective.

        Parameters
        ----------
        update
          The update callback or `None` to reset it.
        args
          Positional arguments for the callback.
        kargs
          Keyword arguments for the callback.

        See Also
        --------
        getUpdate, petsc.TaoSetUpdate

        """
        if update is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (update, args, kargs)
            self.set_attr('__update__', context)
            CHKERR( TaoSetUpdate(self.tao, TAO_Update, <void*>context) )
        else:
            self.set_attr('__update__', None)
            CHKERR( TaoSetUpdate(self.tao, NULL, NULL) )

    def getUpdate(self) -> tuple[TAOUpdateFunction, tuple[Any,...], dict[str, Any]]:
        """Return the callback to compute the update.

        Not collective.

        See Also
        --------
        setUpdate

        """
        return self.get_attr('__update__')

    # --------------

    def computeObjective(self, Vec x) -> float:
        """Compute the value of the objective function.

        Collective.

        Parameters
        ----------
        x
          The parameter vector.

        See Also
        --------
        setObjective, petsc.TaoComputeObjective

        """
        cdef PetscReal f = 0
        CHKERR( TaoComputeObjective(self.tao, x.vec, &f) )
        return toReal(f)

    def computeResidual(self, Vec x, Vec f) -> None:
        """Compute the residual.

        Collective.

        Parameters
        ----------
        x
          The parameter vector.
        f
          The output vector.

        See Also
        --------
        setResidual, petsc.TaoComputeResidual

        """
        CHKERR( TaoComputeResidual(self.tao, x.vec, f.vec) )

    def computeGradient(self, Vec x, Vec g):
        """Compute the gradient of the objective function.

        Collective.

        Parameters
        ----------
        x
          The parameter vector.
        g
          The output gradient vector.

        See Also
        --------
        setGradient, petsc.TaoComputeGradient

        """
        CHKERR( TaoComputeGradient(self.tao, x.vec, g.vec) )

    def computeObjectiveGradient(self, Vec x, Vec g) -> float:
        """Compute the gradient of the objective function and its value.

        Collective.

        Parameters
        ----------
        x
          The parameter vector.
        g
          The output gradient vector.

        See Also
        --------
        setObjectiveGradient, setGradient, setObjective, petsc.TaoComputeObjectiveAndGradient

        """
        cdef PetscReal f = 0
        CHKERR( TaoComputeObjectiveAndGradient(self.tao, x.vec, &f, g.vec) )
        return toReal(f)

    def computeDualVariables(self, Vec xl, Vec xu) -> None:
        """Compute the dual vectors corresponding to variables' bounds.

        Collective.

        See Also
        --------
        petsc.TaoComputeDualVariables

        """
        CHKERR( TaoComputeDualVariables(self.tao, xl.vec, xu.vec) )

    def computeVariableBounds(self, Vec xl, Vec xu) -> None:
        """Compute the vectors corresponding to variables' bounds.

        Collective.

        See Also
        --------
        setVariableBounds, petsc.TaoComputeVariableBounds

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

    def computeConstraints(self, Vec x, Vec c) -> None:
        """Compute the vector corresponding to the constraints.

        Collective.

        Parameters
        ----------
        x
          The parameter vector.
        c
          The output constraints vector.

        See Also
        --------
        setVariableBounds, petsc.TaoComputeVariableBounds

        """
        CHKERR( TaoComputeConstraints(self.tao, x.vec, c.vec) )

    def computeHessian(self, Vec x, Mat H, Mat P=None) -> None:
        """Compute the Hessian of the objective function.

        Collective.

        Parameters
        ----------
        x
          The parameter vector.
        H
          The output Hessian matrix.
        P
          The output Hessian matrix used to construct the preconditioner.

        See Also
        --------
        setHessian, petsc.TaoComputeHessian

        """
        cdef PetscMat hmat = H.mat, pmat = H.mat
        if P is not None: pmat = P.mat
        CHKERR( TaoComputeHessian(self.tao, x.vec, hmat, pmat) )

    def computeJacobian(self, Vec x, Mat J, Mat P=None) -> None:
        """Compute the Jacobian.

        Collective.

        Parameters
        ----------
        x
          The parameter vector.
        J
          The output Jacobian matrix.
        P
          The output Jacobian matrix used to construct the preconditioner.

        See Also
        --------
        setJacobian, petsc.TaoComputeJacobian

        """
        cdef PetscMat jmat = J.mat, pmat = J.mat
        if P is not None: pmat = P.mat
        CHKERR( TaoComputeJacobian(self.tao, x.vec, jmat, pmat) )

    # --------------

    #

    def setTolerances(self, gatol: float = None, grtol: float = None, gttol: float = None) -> None:
        """Set the tolerance parameters used in the solver convergence tests.

        Collective.

        Parameters
        ----------
        gatol
          The absolute norm of the gradient. Defaults to `DEFAULT`.
        grtol
          The relative norm of the gradient with respect to the initial norm of the objective. Defaults to `DEFAULT`.
        gttol
          The relative norm of the gradient with respect to the initial norm of the gradient. Defaults to `DEFAULT`.

        See Also
        --------
        getTolerances, petsc.TaoSetTolerances

        """
        cdef PetscReal _gatol=PETSC_DEFAULT, _grtol=PETSC_DEFAULT, _gttol=PETSC_DEFAULT
        if gatol is not None: _gatol = asReal(gatol)
        if grtol is not None: _grtol = asReal(grtol)
        if gttol is not None: _gttol = asReal(gttol)
        CHKERR( TaoSetTolerances(self.tao, _gatol, _grtol, _gttol) )

    def getTolerances(self) -> tuple[float, float, float]:
        """Return the tolerance parameters used in the solver convergence tests.

        Not collective.

        Returns
        -------
        gatol: float
          The absolute norm of the gradient.
        grtol: float
          The relative norm of the gradient with respect to the initial norm of the objective.
        gttol: float
          The relative norm of the gradient with respect to the initial norm of the gradient.

        See Also
        --------
        setTolerances, petsc.TaoGetTolerances

        """
        cdef PetscReal _gatol=PETSC_DEFAULT, _grtol=PETSC_DEFAULT, _gttol=PETSC_DEFAULT
        CHKERR( TaoGetTolerances(self.tao, &_gatol, &_grtol, &_gttol) )
        return (toReal(_gatol), toReal(_grtol), toReal(_gttol))

    def setMaximumIterations(self, mit: int) -> float:
        """Set the maximum number of solver iterations.

        Collective.

        See Also
        --------
        setTolerances, petsc.TaoSetMaximumIterations

        """
        cdef PetscInt _mit = asInt(mit)
        CHKERR( TaoSetMaximumIterations(self.tao, _mit) )

    def getMaximumIterations(self) -> int:
        """Return the maximum number of solver iterations.

        Not collective.

        See Also
        --------
        setMaximumIterations, petsc.TaoGetMaximumIterations

        """
        cdef PetscInt _mit = PETSC_DEFAULT
        CHKERR( TaoGetMaximumIterations(self.tao, &_mit) )
        return toInt(_mit)

    def setMaximumFunctionEvaluations(self, mit: int) -> None:
        """Set the maximum number of objective evaluations within the solver.

        Collective.

        See Also
        --------
        setMaximumIterations, petsc.TaoSetMaximumFunctionEvaluations

        """
        cdef PetscInt _mit = asInt(mit)
        CHKERR( TaoSetMaximumFunctionEvaluations(self.tao, _mit) )

    def getMaximumFunctionEvaluations(self) -> int:
        """Return the maximum number of objective evaluations within the solver.

        Not collective.

        See Also
        --------
        setMaximumFunctionEvaluations, petsc.TaoGetMaximumFunctionEvaluations

        """
        cdef PetscInt _mit = PETSC_DEFAULT
        CHKERR( TaoGetMaximumFunctionEvaluations(self.tao, &_mit) )
        return toInt(_mit)

    def setConstraintTolerances(self, catol: float = None, crtol: float = None) -> None:
        """Set the constraints tolerance parameters used in the solver convergence tests.

        Collective.

        Parameters
        ----------
        catol
          The absolute norm of the constraints. Defaults to `DEFAULT`.
        crtol
          The relative norm of the constraints. Defaults to `DEFAULT`.

        See Also
        --------
        getConstraintTolerances, petsc.TaoSetConstraintTolerances

        """
        cdef PetscReal _catol=PETSC_DEFAULT, _crtol=PETSC_DEFAULT
        if catol is not None: _catol = asReal(catol)
        if crtol is not None: _crtol = asReal(crtol)
        CHKERR( TaoSetConstraintTolerances(self.tao, _catol, _crtol) )

    def getConstraintTolerances(self) -> tuple[float, float]:
        """Return the constraints tolerance parameters used in the solver convergence tests.

        Not collective.

        Returns
        -------
        catol: float
          The absolute norm of the constraints.
        crtol: float
          The relative norm of the constraints.

        See Also
        --------
        setConstraintTolerances, petsc.TaoGetConstraintTolerances

        """
        cdef PetscReal _catol=PETSC_DEFAULT, _crtol=PETSC_DEFAULT
        CHKERR( TaoGetConstraintTolerances(self.tao, &_catol, &_crtol) )
        return (toReal(_catol), toReal(_crtol))

    def setConvergenceTest(self, converged: TAOConvergedFunction | None, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the callback used to test for solver convergence.

        Logically collective.

        Parameters
        ----------
        converged
          The callback. If `None`, reset to the default convergence test.
        args
          Positional arguments for the callback.
        kargs
          Keyword arguments for the callback.

        See Also
        --------
        getConvergenceTest, petsc.TaoSetConvergenceTest

        """
        if converged is None:
            CHKERR( TaoSetConvergenceTest(self.tao, TaoDefaultConvergenceTest, NULL) )
            self.set_attr('__converged__', None)
        else:
            if args is None: args = ()
            if kargs is None: kargs = {}
            self.set_attr('__converged__', (converged, args, kargs))
            CHKERR( TaoSetConvergenceTest(self.tao, TAO_Converged, NULL) )

    def getConvergenceTest(self) -> tuple[TAOConvergedFunction, tuple[Any, ...], dict[str, Any]]:
        """Return the callback used to test for solver convergence.

        Not collective.

        See Also
        --------
        setConvergenceTest

        """
        return self.get_attr('__converged__')

    def setConvergedReason(self, reason: ConvergedReason) -> None:
        """Set the termination flag.

        Collective.

        See Also
        --------
        getConvergedReason, petsc.TaoSetConvergedReason

        """
        cdef PetscTAOConvergedReason creason = reason
        CHKERR( TaoSetConvergedReason(self.tao, creason) )

    def getConvergedReason(self) -> ConvergedReason:
        """Return the termination flag.

        Not collective.

        See Also
        --------
        setConvergedReason, petsc.TaoGetConvergedReason

        """
        cdef PetscTAOConvergedReason creason = TAO_CONTINUE_ITERATING
        CHKERR( TaoGetConvergedReason(self.tao, &creason) )
        return creason

    def setMonitor(self, monitor: TAOMonitorFunction, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the callback used to monitor solver convergence.

        Logically collective.

        Parameters
        ----------
        monitor
          The callback.
        args
          Positional arguments for the callback.
        kargs
          Keyword arguments for the callback.

        See Also
        --------
        getMonitor, petsc.TaoSetMonitor

        """
        if monitor is None: return
        cdef object monitorlist = self.get_attr('__monitor__')
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        if monitorlist is None:
            CHKERR( TaoSetMonitor(self.tao, TAO_Monitor, NULL, NULL) )
            self.set_attr('__monitor__',  [(monitor, args, kargs)])
        else:
            monitorlist.append((monitor, args, kargs))

    def getMonitor(self) -> list[tuple[TAOMonitorFunction, tuple[Any, ...], dict[str, Any]]]:
        """Return the callback used to monitor solver convergence.

        Not collective.

        See Also
        --------
        setMonitor

        """
        return self.get_attr('__monitor__')

    def cancelMonitor(self) -> None:
        """Cancel all the monitors of the solver.

        Logically collective.

        See Also
        --------
        setMonitor, petsc.TaoCancelMonitors

        """
        CHKERR( TaoCancelMonitors(self.tao) )
        self.set_attr('__monitor__',  None)

    # Tao overwrites these statistics. Copy user defined only if present
    def monitor(self, its: int = None, f: float = None, res: float = None, cnorm: float = None, step: float = None) -> None:
        """Monitor the solver.

        Collective.

        This function should be called without arguments,
        unless users want to modify the values internally stored by the solver.

        Parameters
        ----------
        its
          Current number of iterations or `None` to use the value stored internally by the solver.
        f
          Current value of the objective function or `None` to use the value stored internally by the solver.
        res
          Current value of the residual norm or `None` to use the value stored internally by the solver.
        cnorm
          Current value of the constrains norm or `None` to use the value stored internally by the solver.
        step
          Current value of the step or `None` to use the value stored internally by the solver.

        See Also
        --------
        setMonitor, petsc.TaoMonitor

        """
        cdef PetscInt cits = 0
        cdef PetscReal cf = 0.0
        cdef PetscReal cres = 0.0
        cdef PetscReal ccnorm = 0.0
        cdef PetscReal cstep = 0.0
        CHKERR( TaoGetSolutionStatus(self.tao, &cits, &cf, &cres, &ccnorm, &cstep, NULL) )
        if its is not None:
            cits = asInt(its)
        if f is not None:
            cf = asReal(f)
        if res is not None:
            cres = asReal(res)
        if cnorm is not None:
            ccnorm = asReal(cnorm)
        if step is not None:
            cstep = asReal(step)
        CHKERR( TaoMonitor(self.tao, cits, cf, cres, ccnorm, cstep) )

    #

    def solve(self, Vec x=None) -> None:
        """Solve the optimization problem.

        Collective.

        Parameters
        ----------
        x
          the starting vector or `None` to use the vector stored internally.

        See Also
        --------
        setSolution, getSolution, petsc.TaoSolve

        """
        if x is not None:
            CHKERR( TaoSetSolution(self.tao, x.vec) )
        CHKERR( TaoSolve(self.tao) )

    def getSolution(self) -> Vec:
        """Return the vector holding the solution.

        Not collective.

        See Also
        --------
        setSolution, petsc.TaoGetSolution

        """
        cdef Vec vec = Vec()
        CHKERR( TaoGetSolution(self.tao, &vec.vec) )
        PetscINCREF(vec.obj)
        return vec

    def setGradientNorm(self, Mat mat) -> None:
        """Set the matrix used to compute inner products.

        Collective.

        See Also
        --------
        getGradientNorm, petsc.TaoSetGradientNorm

        """
        CHKERR( TaoSetGradientNorm(self.tao, mat.mat) )

    def getGradientNorm(self) -> Mat:
        """Return the matrix used to compute inner products.

        Not collective.

        See Also
        --------
        setGradientNorm, petsc.TaoGetGradientNorm

        """
        cdef Mat mat = Mat()
        CHKERR( TaoGetGradientNorm(self.tao, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

    def setLMVMH0(self, Mat mat) -> None:
        """Set the initial Hessian for the quasi-Newton approximation.

        Collective.

        See Also
        --------
        getLMVMH0, petsc.TaoLMVMSetH0

        """
        CHKERR( TaoLMVMSetH0(self.tao, mat.mat) )

    def getLMVMH0(self) -> Mat:
        """Return the initial Hessian for the quasi-Newton approximation.

        Not collective.

        See Also
        --------
        setLMVMH0, petsc.TaoLMVMGetH0

        """
        cdef Mat mat = Mat()
        CHKERR( TaoLMVMGetH0(self.tao, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

    def getLMVMH0KSP(self) -> KSP:
        """Return the linear solver for applying the inverse of the initial Hessian approximation.

        Not collective.

        See Also
        --------
        setLMVMH0, petsc.TaoLMVMGetH0KSP

        """
        cdef KSP ksp = KSP()
        CHKERR( TaoLMVMGetH0KSP(self.tao, &ksp.ksp) )
        PetscINCREF(ksp.obj)
        return ksp

    def getVariableBounds(self) -> tuple[Vec, Vec]:
        """Return the upper and lower bounds vectors.

        Not collective.

        See Also
        --------
        setVariableBounds, petsc.TaoGetVariableBounds

        """
        cdef Vec xl = Vec(), xu = Vec()
        CHKERR( TaoGetVariableBounds(self.tao, &xl.vec, &xu.vec) )
        PetscINCREF(xl.obj); PetscINCREF(xu.obj)
        return (xl, xu)

    def setIterationNumber(self, its: int) -> None:
        """Set the current iteration number.

        Collective.

        See Also
        --------
        getIterationNumber, petsc.TaoSetIterationNumber

        """
        cdef PetscInt ival = asInt(its)
        CHKERR( TaoSetIterationNumber(self.tao, ival) )

    def getIterationNumber(self) -> int:
        """Return the current iteration number.

        Not collective.

        See Also
        --------
        setIterationNumber, petsc.TaoGetIterationNumber

        """
        cdef PetscInt its=0
        CHKERR( TaoGetIterationNumber(self.tao, &its) )
        return toInt(its)

    def getObjectiveValue(self) -> float:
        """Return the current value of the objective function.

        Not collective.

        See Also
        --------
        setObjective, petsc.TaoGetSolutionStatus

        """
        cdef PetscReal fval=0
        CHKERR( TaoGetSolutionStatus(self.tao, NULL, &fval, NULL, NULL, NULL, NULL) )
        return toReal(fval)

    getFunctionValue = getObjectiveValue

    def getConvergedReason(self) -> ConvergedReason:
        """Return the reason for the solver convergence.

        Not collective.

        See Also
        --------
        petsc.TaoGetConvergedReason

        """
        cdef PetscTAOConvergedReason reason = TAO_CONTINUE_ITERATING
        CHKERR( TaoGetConvergedReason(self.tao, &reason) )
        return reason

    def getSolutionNorm(self) -> tuple[float, float, float]:
        """Return the value of the objective function, the norm of the gradient and the norm of the constraints.

        Not collective.

        Returns
        -------
        f: float
          Current value of the objective function.
        res: float
          Current value of the residual norm.
        cnorm: float
          Current value of the constrains norm.

        See Also
        --------
        getSolutionStatus, petsc.TaoGetSolutionStatus

        """
        cdef PetscReal gnorm=0
        cdef PetscReal cnorm=0
        cdef PetscReal fval=0
        CHKERR( TaoGetSolutionStatus(self.tao, NULL, &fval, &gnorm, &cnorm, NULL, NULL) )
        return (toReal(fval), toReal(gnorm), toReal(cnorm))

    def getSolutionStatus(self) -> tuple[int, float, float, float, float, ConvergedReason]:
        """Return the solution status.

        Not collective.

        Returns
        -------
        its: int
          Current number of iterations.
        f: float
          Current value of the objective function.
        res: float
          Current value of the residual norm.
        cnorm: float
          Current value of the constrains norm.
        step: float
          Current value of the step.
        reason: ConvergedReason
          Current value of converged reason.

        See Also
        --------
        petsc.TaoGetSolutionStatus

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

    def getKSP(self) -> KSP:
        """Return the linear solver used by the nonlinear solver.

        Not collective.

        See Also
        --------
        petsc.TaoGetKSP

        """
        cdef KSP ksp = KSP()
        CHKERR( TaoGetKSP(self.tao, &ksp.ksp) )
        PetscINCREF(ksp.obj)
        return ksp

    # BRGN routines

    def getBRGNSubsolver(self) -> TAO:
        """Return the subsolver inside the BRGN solver.

        Not collective.

        See Also
        --------
        petsc.TaoBRGNGetSubsolver

        """
        cdef TAO subsolver = TAO()
        CHKERR( TaoBRGNGetSubsolver(self.tao, &subsolver.tao) )
        PetscINCREF(subsolver.obj)
        return subsolver

    def setBRGNRegularizerObjectiveGradient(self, objgrad, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute the regularizer objective and gradient.

        Logically collective.

        See Also
        --------
        petsc.TaoBRGNSetRegularizerObjectiveAndGradientRoutine

        """
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (objgrad, args, kargs)
        self.set_attr("__brgnregobjgrad__", context)
        CHKERR( TaoBRGNSetRegularizerObjectiveAndGradientRoutine(self.tao, TAO_BRGNRegObjGrad, <void*>context) )

    def setBRGNRegularizerHessian(self, hessian, Mat H=None, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute the regularizer Hessian.

        Logically collective.

        See Also
        --------
        petsc.TaoBRGNSetRegularizerHessianRoutine

        """
        cdef PetscMat Hmat = NULL
        if H is not None: Hmat = H.mat
        if args is None: args = ()
        if kargs is None: kargs = {}
        context = (hessian, args, kargs)
        self.set_attr("__brgnreghessian__", context)
        CHKERR( TaoBRGNSetRegularizerHessianRoutine(self.tao, Hmat, TAO_BRGNRegHessian, <void*>context) )

    def setBRGNRegularizerWeight(self, weight: float) -> None:
        """Set the regularizer weight.

        Collective.

        """
        cdef PetscReal cweight = asReal(weight)
        CHKERR( TaoBRGNSetRegularizerWeight(self.tao, cweight) )

    def setBRGNSmoothL1Epsilon(self, epsilon: float) -> None:
        """Set the smooth L1 epsilon.

        Collective.

        See Also
        --------
        petsc.TaoBRGNSetL1SmoothEpsilon

        """
        cdef PetscReal ceps = asReal(epsilon)
        CHKERR( TaoBRGNSetL1SmoothEpsilon(self.tao, ceps) )

    def setBRGNDictionaryMatrix(self, Mat D) -> None:
        """Set the dictionary matrix.

        Collective.

        See Also
        --------
        petsc.TaoBRGNSetDictionaryMatrix

        """
        CHKERR( TaoBRGNSetDictionaryMatrix(self.tao, D.mat) )

    def getBRGNDampingVector(self) -> Vec:
        """Return the damping vector.

        Not collective.

        """
        #FIXME
        #See Also
        #--------
        #petsc.TaoBRGNGetDampingVector
        cdef Vec damp = Vec()
        CHKERR( TaoBRGNGetDampingVector(self.tao, &damp.vec) )
        PetscINCREF(damp.obj)
        return damp

    def createPython(self, context: Any = None, comm: Comm | None = None) -> Self:
        """Create an optimization solver of Python type.

        Collective.

        Parameters
        ----------
        context
          An instance of the Python class implementing the required methods.
        comm
          The communicator associated with the object. Defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc_python_tao, setType, setPythonContext, TAO.Type.PYTHON

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscTAO tao = NULL
        CHKERR( TaoCreate(ccomm, &tao) )
        PetscCLEAR(self.obj); self.tao = tao
        CHKERR( TaoSetType(self.tao, TAOPYTHON) )
        CHKERR( TaoPythonSetContext(self.tao, <void*>context) )
        return self

    def setPythonContext(self, context: Any) -> None:
        """Set the instance of the Python class implementing the required Python methods.

        Not collective.

        See Also
        --------
        petsc_python_tao, getPythonContext

        """
        CHKERR( TaoPythonSetContext(self.tao, <void*>context) )

    def getPythonContext(self) -> Any:
        """Return the instance of the Python class implementing the required Python methods.

        Not collective.

        See Also
        --------
        petsc_python_tao, setPythonContext

        """
        cdef void *context = NULL
        CHKERR( TaoPythonGetContext(self.tao, &context) )
        if context == NULL: return None
        else: return <object> context

    def setPythonType(self, py_type: str) -> None:
        """Set the fully qualified Python name of the class to be used.

        Collective.

        See Also
        --------
        petsc_python_tao, setPythonContext, getPythonType, petsc.TaoPythonSetType

        """
        cdef const char *cval = NULL
        py_type = str2bytes(py_type, &cval)
        CHKERR( TaoPythonSetType(self.tao, cval) )

    def getPythonType(self):
        """Return the full dotted Python name of the class used by the solver.

        Not collective.

        See Also
        --------
        petsc_python_tao, setPythonContext, setPythonType, petsc.TaoPythonGetType

        """
        cdef const char *cval = NULL
        CHKERR( TaoPythonGetType(self.tao, &cval) )
        return bytes2str(cval)

    # --- backward compatibility ---

    setInitial = setSolution

    # --- application context ---

    property appctx:
        """Application context."""
        def __get__(self) -> Any:
            return self.getAppCtx()
        def __set__(self, value: Any):
            self.setAppCtx(value)

    # --- linear solver ---

    property ksp:
        """Linear solver."""
        def __get__(self) -> KSP:
            return self.getKSP()

    # --- tolerances ---

    # FIXME: tolerances all broken
    property ftol:
        """Broken."""
        def __get__(self) -> Any:
            return self.getFunctionTolerances()
        def __set__(self, value):
            if isinstance(value, (tuple, list)):
                self.setFunctionTolerances(*value)
            elif isinstance(value, dict):
                self.setFunctionTolerances(**value)
            else:
                raise TypeError("expecting tuple/list or dict")

    property gtol:
        """Broken."""
        def __get__(self) -> Any:
            return self.getGradientTolerances()
        def __set__(self, value):
            if isinstance(value, (tuple, list)):
                self.getGradientTolerances(*value)
            elif isinstance(value, dict):
                self.getGradientTolerances(**value)
            else:
                raise TypeError("expecting tuple/list or dict")

    property ctol:
        """Broken."""
        def __get__(self) -> Any:
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
        """Number of iterations."""
        def __get__(self) -> int:
            return self.getIterationNumber()

    property gnorm:
        """Gradient norm."""
        def __get__(self) -> float:
            return self.getSolutionNorm()[1]

    property cnorm:
        """Constraints norm."""
        def __get__(self) -> float:
            return self.getSolutionNorm()[2]

    property solution:
        """Solution vector."""
        def __get__(self) -> Vec:
            return self.getSolution()

    property objective:
        """Objective value."""
        def __get__(self) -> float:
            return self.getObjectiveValue()

    property function:
        """Objective value."""
        def __get__(self) -> float:
            return self.getFunctionValue()

    property gradient:
        """Gradient vector."""
        def __get__(self) -> Vec:
            return self.getGradient()[0]

    # --- convergence ---

    property reason:
        """Converged reason."""
        def __get__(self) -> ConvergedReason:
            return self.getConvergedReason()

    property iterating:
        """Boolean indicating if the solver is not converged yet."""
        def __get__(self) -> bool:
            return self.reason == 0

    property converged:
        """Boolean indicating if the solver has converged."""
        def __get__(self) -> bool:
            return self.reason > 0

    property diverged:
        """Boolean indicating if the solver has failed."""
        def __get__(self) -> bool:
            return self.reason < 0

# --------------------------------------------------------------------

del TAOType
del TAOConvergedReason

# --------------------------------------------------------------------
