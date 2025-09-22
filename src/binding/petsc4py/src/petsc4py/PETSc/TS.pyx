
# -----------------------------------------------------------------------------

class TSType(object):
    """The time stepping method."""
    # native
    EULER           = S_(TSEULER)
    BEULER          = S_(TSBEULER)
    BASICSYMPLECTIC = S_(TSBASICSYMPLECTIC)
    PSEUDO          = S_(TSPSEUDO)
    CN              = S_(TSCN)
    SUNDIALS        = S_(TSSUNDIALS)
    RK              = S_(TSRK)
    PYTHON          = S_(TSPYTHON)
    THETA           = S_(TSTHETA)
    ALPHA           = S_(TSALPHA)
    ALPHA2          = S_(TSALPHA2)
    GLLE            = S_(TSGLLE)
    GLEE            = S_(TSGLEE)
    SSP             = S_(TSSSP)
    ARKIMEX         = S_(TSARKIMEX)
    DIRK            = S_(TSDIRK)
    ROSW            = S_(TSROSW)
    EIMEX           = S_(TSEIMEX)
    MIMEX           = S_(TSMIMEX)
    BDF             = S_(TSBDF)
    RADAU5          = S_(TSRADAU5)
    MPRK            = S_(TSMPRK)
    DISCGRAD        = S_(TSDISCGRAD)
    # aliases
    FE = EULER
    BE = BEULER
    TH = THETA
    CRANK_NICOLSON = CN
    RUNGE_KUTTA    = RK


class TSRKType(object):
    """The *RK* subtype."""
    RK1FE = S_(TSRK1FE)
    RK2A  = S_(TSRK2A)
    RK2B  = S_(TSRK2B)
    RK4   = S_(TSRK4)
    RK3BS = S_(TSRK3BS)
    RK3   = S_(TSRK3)
    RK5F  = S_(TSRK5F)
    RK5DP = S_(TSRK5DP)
    RK5BS = S_(TSRK5BS)
    RK6VR = S_(TSRK6VR)
    RK7VR = S_(TSRK7VR)
    RK8VR = S_(TSRK8VR)


class TSARKIMEXType(object):
    """The *ARKIMEX* subtype."""
    ARKIMEX1BEE   = S_(TSARKIMEX1BEE)
    ARKIMEXA2     = S_(TSARKIMEXA2)
    ARKIMEXL2     = S_(TSARKIMEXL2)
    ARKIMEXARS122 = S_(TSARKIMEXARS122)
    ARKIMEX2C     = S_(TSARKIMEX2C)
    ARKIMEX2D     = S_(TSARKIMEX2D)
    ARKIMEX2E     = S_(TSARKIMEX2E)
    ARKIMEXPRSSP2 = S_(TSARKIMEXPRSSP2)
    ARKIMEX3      = S_(TSARKIMEX3)
    ARKIMEXBPR3   = S_(TSARKIMEXBPR3)
    ARKIMEXARS443 = S_(TSARKIMEXARS443)
    ARKIMEX4      = S_(TSARKIMEX4)
    ARKIMEX5      = S_(TSARKIMEX5)


class TSDIRKType(object):
    """The *DIRK* subtype."""
    DIRKS212      = S_(TSDIRKS212)
    DIRKES122SAL  = S_(TSDIRKES122SAL)
    DIRKES213SAL  = S_(TSDIRKES213SAL)
    DIRKES324SAL  = S_(TSDIRKES324SAL)
    DIRKES325SAL  = S_(TSDIRKES325SAL)
    DIRK657A      = S_(TSDIRK657A)
    DIRKES648SA   = S_(TSDIRKES648SA)
    DIRK658A      = S_(TSDIRK658A)
    DIRKS659A     = S_(TSDIRKS659A)
    DIRK7510SAL   = S_(TSDIRK7510SAL)
    DIRKES7510SA  = S_(TSDIRKES7510SA)
    DIRK759A      = S_(TSDIRK759A)
    DIRKS7511SAL  = S_(TSDIRKS7511SAL)
    DIRK8614A     = S_(TSDIRK8614A)
    DIRK8616SAL   = S_(TSDIRK8616SAL)
    DIRKES8516SAL = S_(TSDIRKES8516SAL)


class TSProblemType(object):
    """Distinguishes linear and nonlinear problems."""
    LINEAR    = TS_LINEAR
    NONLINEAR = TS_NONLINEAR


class TSEquationType(object):
    """Distinguishes among types of explicit and implicit equations."""
    UNSPECIFIED               = TS_EQ_UNSPECIFIED
    EXPLICIT                  = TS_EQ_EXPLICIT
    ODE_EXPLICIT              = TS_EQ_ODE_EXPLICIT
    DAE_SEMI_EXPLICIT_INDEX1  = TS_EQ_DAE_SEMI_EXPLICIT_INDEX1
    DAE_SEMI_EXPLICIT_INDEX2  = TS_EQ_DAE_SEMI_EXPLICIT_INDEX2
    DAE_SEMI_EXPLICIT_INDEX3  = TS_EQ_DAE_SEMI_EXPLICIT_INDEX3
    DAE_SEMI_EXPLICIT_INDEXHI = TS_EQ_DAE_SEMI_EXPLICIT_INDEXHI
    IMPLICIT                  = TS_EQ_IMPLICIT
    ODE_IMPLICIT              = TS_EQ_ODE_IMPLICIT
    DAE_IMPLICIT_INDEX1       = TS_EQ_DAE_IMPLICIT_INDEX1
    DAE_IMPLICIT_INDEX2       = TS_EQ_DAE_IMPLICIT_INDEX2
    DAE_IMPLICIT_INDEX3       = TS_EQ_DAE_IMPLICIT_INDEX3
    DAE_IMPLICIT_INDEXHI      = TS_EQ_DAE_IMPLICIT_INDEXHI


class TSExactFinalTime(object):
    """The method for ending time stepping."""
    UNSPECIFIED = TS_EXACTFINALTIME_UNSPECIFIED
    STEPOVER    = TS_EXACTFINALTIME_STEPOVER
    INTERPOLATE = TS_EXACTFINALTIME_INTERPOLATE
    MATCHSTEP   = TS_EXACTFINALTIME_MATCHSTEP


class TSConvergedReason:
    """The reason the time step is converging."""
    # iterating
    CONVERGED_ITERATING      = TS_CONVERGED_ITERATING
    ITERATING                = TS_CONVERGED_ITERATING
    # converged
    CONVERGED_TIME           = TS_CONVERGED_TIME
    CONVERGED_ITS            = TS_CONVERGED_ITS
    CONVERGED_USER           = TS_CONVERGED_USER
    CONVERGED_EVENT          = TS_CONVERGED_EVENT
    # diverged
    DIVERGED_NONLINEAR_SOLVE = TS_DIVERGED_NONLINEAR_SOLVE
    DIVERGED_STEP_REJECTED   = TS_DIVERGED_STEP_REJECTED

# -----------------------------------------------------------------------------


cdef class TS(Object):
    """ODE integrator.

    TS is described in the `PETSc manual <petsc:manual/ts>`.

    See Also
    --------
    petsc.TS

    """
    Type = TSType
    RKType = TSRKType
    ARKIMEXType = TSARKIMEXType
    DIRKType = TSDIRKType
    ProblemType = TSProblemType
    EquationType = TSEquationType
    ExactFinalTime = TSExactFinalTime
    ExactFinalTimeOption = TSExactFinalTime
    ConvergedReason = TSConvergedReason

    # --- xxx ---

    def __cinit__(self):
        self.obj = <PetscObject*> &self.ts
        self.ts = NULL

    # --- xxx ---

    def view(self, Viewer viewer=None) -> None:
        """Print the `TS` object.

        Collective.

        Parameters
        ----------
        viewer
            The visualization context.

        Notes
        -----
        ``-ts_view`` calls TSView at the end of TSStep

        See Also
        --------
        petsc.TSView

        """
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR(TSView(self.ts, cviewer))

    def load(self, Viewer viewer) -> None:
        """Load a `TS` that has been stored in binary with `view`.

        Collective.

        Parameters
        ----------
        viewer
            The visualization context.

        See Also
        --------
        petsc.TSLoad

        """
        CHKERR(TSLoad(self.ts, viewer.vwr))

    def destroy(self) -> Self:
        """Destroy the `TS` that was created with `create`.

        Collective.

        See Also
        --------
        petsc.TSDestroy

        """
        CHKERR(TSDestroy(&self.ts))
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create an empty `TS`.

        Collective.

        The problem type can then be set with `setProblemType` and the type of
        solver can then be set with `setType`.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.TSCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscTS newts = NULL
        CHKERR(TSCreate(ccomm, &newts))
        CHKERR(PetscCLEAR(self.obj)); self.ts = newts
        return self

    def clone(self) -> TS:
        """Return a shallow clone of the `TS` object.

        Collective.

        See Also
        --------
        petsc.TSClone

        """
        cdef TS ts = TS()
        CHKERR(TSClone(self.ts, &ts.ts))
        return ts

    def setType(self, ts_type: Type | str) -> None:
        """Set the method to be used as the `TS` solver.

        Collective.

        Parameters
        ----------
        ts_type
            The solver type.

        Notes
        -----
        ``-ts_type`` sets the method from the commandline

        See Also
        --------
        petsc.TSSetType

        """
        cdef PetscTSType cval = NULL
        ts_type = str2bytes(ts_type, &cval)
        CHKERR(TSSetType(self.ts, cval))

    def setRKType(self, ts_type: RKType | str) -> None:
        """Set the type of the *Runge-Kutta* scheme.

        Logically collective.

        Parameters
        ----------
        ts_type
            The type of scheme.

        Notes
        -----
            ``-ts_rk_type`` sets scheme type from the commandline.

        See Also
        --------
        petsc.TSRKSetType

        """
        cdef PetscTSRKType cval = NULL
        ts_type = str2bytes(ts_type, &cval)
        CHKERR(TSRKSetType(self.ts, cval))

    def setARKIMEXType(self, ts_type: ARKIMEXType | str) -> None:
        """Set the type of `Type.ARKIMEX` scheme.

        Logically collective.

        Parameters
        ----------
        ts_type
            The type of `Type.ARKIMEX` scheme.

        Notes
        -----
            ``-ts_arkimex_type`` sets scheme type from the commandline.

        See Also
        --------
        petsc.TSARKIMEXSetType

        """
        cdef PetscTSARKIMEXType cval = NULL
        ts_type = str2bytes(ts_type, &cval)
        CHKERR(TSARKIMEXSetType(self.ts, cval))

    def setARKIMEXFullyImplicit(self, flag: bool) -> None:
        """Solve both parts of the equation implicitly.

        Logically collective.

        Parameters
        ----------
        flag
            Set to True for fully implicit.

        See Also
        --------
        petsc.TSARKIMEXSetFullyImplicit

        """
        cdef PetscBool bval = asBool(flag)
        CHKERR(TSARKIMEXSetFullyImplicit(self.ts, bval))

    def setARKIMEXFastSlowSplit(self, flag: bool) -> None:
        """Use ARKIMEX for solving a fast-slow system.

        Logically collective.

        Parameters
        ----------
        flag
            Set to True for fast-slow partitioned systems.

        See Also
        --------
        petsc.TSARKIMEXSetType
        """
        cdef PetscBool bval = asBool(flag)
        CHKERR(TSARKIMEXSetFastSlowSplit(self.ts, bval))

    def getType(self) -> str:
        """Return the `TS` type.

        Not collective.

        See Also
        --------
        petsc.TSGetType

        """
        cdef PetscTSType cval = NULL
        CHKERR(TSGetType(self.ts, &cval))
        return bytes2str(cval)

    def getRKType(self) -> str:
        """Return the `Type.RK` scheme.

        Not collective.

        See Also
        --------
        petsc.TSRKGetType

        """
        cdef PetscTSRKType cval = NULL
        CHKERR(TSRKGetType(self.ts, &cval))
        return bytes2str(cval)

    def getARKIMEXType(self) -> str:
        """Return the `Type.ARKIMEX` scheme.

        Not collective.

        See Also
        --------
        petsc.TSARKIMEXGetType

        """
        cdef PetscTSARKIMEXType cval = NULL
        CHKERR(TSARKIMEXGetType(self.ts, &cval))
        return bytes2str(cval)

    def setDIRKType(self, ts_type: DIRKType | str) -> None:
        """Set the type of `Type.DIRK` scheme.

        Logically collective.

        Parameters
        ----------
        ts_type
            The type of `Type.DIRK` scheme.

        Notes
        -----
            ``-ts_dirk_type`` sets scheme type from the commandline.

        See Also
        --------
        petsc.TSDIRKSetType

        """
        cdef PetscTSDIRKType cval = NULL
        ts_type = str2bytes(ts_type, &cval)
        CHKERR(TSDIRKSetType(self.ts, cval))

    def getDIRKType(self) -> str:
        """Return the `Type.DIRK` scheme.

        Not collective.

        See Also
        --------
        setDIRKType, petsc.TSDIRKGetType

        """
        cdef PetscTSDIRKType cval = NULL
        CHKERR(TSDIRKGetType(self.ts, &cval))
        return bytes2str(cval)

    def setProblemType(self, ptype: ProblemType) -> None:
        """Set the type of problem to be solved.

        Logically collective.

        Parameters
        ----------
        ptype
            The type of problem of the forms.

        See Also
        --------
        petsc.TSSetProblemType

        """
        CHKERR(TSSetProblemType(self.ts, ptype))

    def getProblemType(self) -> ProblemType:
        """Return the type of problem to be solved.

        Not collective.

        See Also
        --------
        petsc.TSGetProblemType

        """
        cdef PetscTSProblemType ptype = TS_NONLINEAR
        CHKERR(TSGetProblemType(self.ts, &ptype))
        return ptype

    def setEquationType(self, eqtype: EquationType) -> None:
        """Set the type of the equation that `TS` is solving.

        Logically collective.

        Parameters
        ----------
        eqtype
            The type of equation.

        See Also
        --------
        petsc.TSSetEquationType

        """
        CHKERR(TSSetEquationType(self.ts, eqtype))

    def getEquationType(self) -> EquationType:
        """Get the type of the equation that `TS` is solving.

        Not collective.

        See Also
        --------
        petsc.TSGetEquationType

        """
        cdef PetscTSEquationType eqtype = TS_EQ_UNSPECIFIED
        CHKERR(TSGetEquationType(self.ts, &eqtype))
        return eqtype

    def setOptionsPrefix(self, prefix : str | None) -> None:
        """Set the prefix used for all the `TS` options.

        Logically collective.

        Parameters
        ----------
        prefix
            The prefix to prepend to all option names.

        Notes
        -----
        A hyphen must not be given at the beginning of the prefix name.

        See Also
        --------
        petsc_options, petsc.TSSetOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR(TSSetOptionsPrefix(self.ts, cval))

    def getOptionsPrefix(self) -> str:
        """Return the prefix used for all the `TS` options.

        Not collective.

        See Also
        --------
        petsc.TSGetOptionsPrefix

        """
        cdef const char *cval = NULL
        CHKERR(TSGetOptionsPrefix(self.ts, &cval))
        return bytes2str(cval)

    def appendOptionsPrefix(self, prefix: str | None) -> None:
        """Append to the prefix used for all the `TS` options.

        Logically collective.

        Parameters
        ----------
        prefix
            The prefix to append to the current prefix.

        Notes
        -----
        A hyphen must not be given at the beginning of the prefix name.

        See Also
        --------
        petsc_options, petsc.TSAppendOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR(TSAppendOptionsPrefix(self.ts, cval))

    def setFromOptions(self) -> None:
        """Set various `TS` parameters from user options.

        Collective.

        See Also
        --------
        petsc_options, petsc.TSSetFromOptions

        """
        CHKERR(TSSetFromOptions(self.ts))

    # --- application context ---

    def setAppCtx(self, appctx: Any) -> None:
        """Set the application context.

        Not collective.

        Parameters
        ----------
        appctx
            The application context.

        """
        self.set_attr('__appctx__', appctx)

    def getAppCtx(self) -> Any:
        """Return the application context."""
        return self.get_attr('__appctx__')

    # --- user RHS Function/Jacobian routines ---

    def setRHSFunction(
        self,
        function: TSRHSFunction | None,
        Vec f=None,
        args : tuple[Any, ...] | None = None,
        kargs : dict[str, Any] | None = None) -> None:
        """Set the routine for evaluating the function ``G`` in ``U_t = G(t, u)``.

        Logically collective.

        Parameters
        ----------
        function
            The right-hand side function.
        f
            The vector into which the right-hand side is computed.
        args
            Additional positional arguments for ``function``.
        kargs
            Additional keyword arguments for ``function``.

        See Also
        --------
        petsc.TSSetRHSFunction

        """
        cdef PetscVec fvec=NULL
        if f is not None: fvec = f.vec
        if function is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (function, args, kargs)
            self.set_attr('__rhsfunction__', context)
            CHKERR(TSSetRHSFunction(self.ts, fvec, TS_RHSFunction, <void*>context))
        else:
            CHKERR(TSSetRHSFunction(self.ts, fvec, NULL, NULL))

    def setRHSJacobian(
        self,
        jacobian: TSRHSJacobian | None,
        Mat J=None,
        Mat P=None,
        args : tuple[Any, ...] | None = None,
        kargs : dict[str, Any] | None = None) -> None:
        """Set the function to compute the Jacobian of ``G`` in ``U_t = G(U, t)``.

        Logically collective.

        Parameters
        ----------
        jacobian
            The right-hand side function.
        J
            The matrix into which the jacobian is computed.
        P
            The matrix into which the preconditioner is computed.
        args
            Additional positional arguments for ``jacobian``.
        kargs
            Additional keyword arguments for ``jacobian``.

        See Also
        --------
        petsc.TSSetRHSJacobian

        """
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat=Jmat
        if P is not None: Pmat = P.mat
        if jacobian is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (jacobian, args, kargs)
            self.set_attr('__rhsjacobian__', context)
            CHKERR(TSSetRHSJacobian(self.ts, Jmat, Pmat, TS_RHSJacobian, <void*>context))
        else:
            CHKERR(TSSetRHSJacobian(self.ts, Jmat, Pmat, NULL, NULL))

    def computeRHSFunction(self, t: float, Vec x, Vec f) -> None:
        """Evaluate the right-hand side function.

        Collective.

        Parameters
        ----------
        t
            The time at which to evaluate the RHS.
        x
            The state vector.
        f
            The Vec into which the RHS is computed.

        See Also
        --------
        petsc.TSComputeRHSFunction

        """
        cdef PetscReal time = asReal(t)
        CHKERR(TSComputeRHSFunction(self.ts, time, x.vec, f.vec))

    def computeRHSFunctionLinear(self, t: float, Vec x, Vec f) -> None:
        """Evaluate the right-hand side via the user-provided Jacobian.

        Collective.

        Parameters
        ----------
        t
            The time at which to evaluate the RHS.
        x
            The state vector.
        f
            The Vec into which the RHS is computed.

        See Also
        --------
        petsc.TSComputeRHSFunctionLinear

        """
        cdef PetscReal time = asReal(t)
        CHKERR(TSComputeRHSFunctionLinear(self.ts, time, x.vec, f.vec, NULL))

    def computeRHSJacobian(self, t: float, Vec x, Mat J, Mat P=None) -> None:
        """Compute the Jacobian matrix that has been set with `setRHSJacobian`.

        Collective.

        Parameters
        ----------
        t
            The time at which to evaluate the Jacobian.
        x
            The state vector.
        J
            The matrix into which the Jacobian is computed.
        P
            The optional matrix to use for building a preconditioner.

        See Also
        --------
        petsc.TSComputeRHSJacobian

        """
        cdef PetscReal time = asReal(t)
        cdef PetscMat jmat = J.mat, pmat = J.mat
        if P is not None: pmat = P.mat
        CHKERR(TSComputeRHSJacobian(self.ts, time, x.vec, jmat, pmat))

    def computeRHSJacobianConstant(self, t: float, Vec x, Mat J, Mat P=None) -> None:
        """Reuse a Jacobian that is time-independent.

        Collective.

        Parameters
        ----------
        t
            The time at which to evaluate the Jacobian.
        x
            The state vector.
        J
            A pointer to the stored Jacobian.
        P
            An optional pointer to the matrix used to construct the preconditioner.

        See Also
        --------
        petsc.TSComputeRHSJacobianConstant

        """
        cdef PetscReal time = asReal(t)
        cdef PetscMat jmat = J.mat, pmat = J.mat
        if P is not None: pmat = P.mat
        CHKERR(TSComputeRHSJacobianConstant(self.ts, time, x.vec, jmat, pmat, NULL))

    def getRHSFunction(self) -> tuple[Vec, TSRHSFunction]:
        """Return the vector where the rhs is stored and the function used to compute it.

        Not collective.

        See Also
        --------
        petsc.TSGetRHSFunction

        """
        cdef Vec f = Vec()
        CHKERR(TSGetRHSFunction(self.ts, &f.vec, NULL, NULL))
        CHKERR(PetscINCREF(f.obj))
        cdef object function = self.get_attr('__rhsfunction__')
        return (f, function)

    def getRHSJacobian(self) -> tuple[Mat, Mat, TSRHSJacobian]:
        """Return the Jacobian and the function used to compute them.

        Not collective.

        See Also
        --------
        petsc.TSGetRHSJacobian

        """
        cdef Mat J = Mat(), P = Mat()
        CHKERR(TSGetRHSJacobian(self.ts, &J.mat, &P.mat, NULL, NULL))
        CHKERR(PetscINCREF(J.obj)); CHKERR(PetscINCREF(P.obj))
        cdef object jacobian = self.get_attr('__rhsjacobian__')
        return (J, P, jacobian)

    # --- user Implicit Function/Jacobian routines ---

    def setIFunction(
        self,
        function: TSIFunction | None,
        Vec f=None,
        args : tuple[Any, ...] | None = None,
        kargs : dict[str, Any] | None = None) -> None:
        """Set the function representing the DAE to be solved.

        Logically collective.

        Parameters
        ----------
        function
            The right-hand side function.
        f
            The vector to store values or `None` to be created internally.
        args
            Additional positional arguments for ``function``.
        kargs
            Additional keyword arguments for ``function``.

        See Also
        --------
        petsc.TSSetIFunction

        """
        cdef PetscVec fvec=NULL
        if f is not None: fvec = f.vec
        if function is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (function, args, kargs)
            self.set_attr('__ifunction__', context)
            CHKERR(TSSetIFunction(self.ts, fvec, TS_IFunction, <void*>context))
        else:
            CHKERR(TSSetIFunction(self.ts, fvec, NULL, NULL))

    def setIJacobian(
        self,
        jacobian: TSIJacobian | None,
        Mat J=None,
        Mat P=None,
        args : tuple[Any, ...] | None = None,
        kargs : dict[str, Any] | None = None) -> None:
        """Set the function to compute the Jacobian.

        Logically collective.

        Set the function to compute the matrix ``dF/dU + a*dF/dU_t`` where
        ``F(t, U, U_t)`` is the function provided with `setIFunction`.

        Parameters
        ----------
        jacobian
            The function which computes the Jacobian.
        J
            The matrix into which the Jacobian is computed.
        P
            The optional matrix to use for building a preconditioner matrix.
        args
            Additional positional arguments for ``jacobian``.
        kargs
            Additional keyword arguments for ``jacobian``.

        See Also
        --------
        petsc.TSSetIJacobian

        """
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat=Jmat
        if P is not None: Pmat = P.mat
        if jacobian is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (jacobian, args, kargs)
            self.set_attr('__ijacobian__', context)
            CHKERR(TSSetIJacobian(self.ts, Jmat, Pmat, TS_IJacobian, <void*>context))
        else:
            CHKERR(TSSetIJacobian(self.ts, Jmat, Pmat, NULL, NULL))

    def setIJacobianP(
        self,
        jacobian,
        Mat J=None,
        args : tuple[Any, ...] | None = None,
        kargs : dict[str, Any] | None = None) -> None:
        """Set the function that computes the Jacobian.

        Logically collective.

        Set the function that computes the Jacobian of ``F`` with respect to
        the parameters ``P`` where ``F(Udot, U, t) = G(U, P, t)``, as well as the
        location to store the matrix.

        Parameters
        ----------
        jacobian
            The function which computes the Jacobian.
        J
            The matrix into which the Jacobian is computed.
        args
            Additional positional arguments for ``jacobian``.
        kargs
            Additional keyword arguments for ``jacobian``.

        See Also
        --------
        petsc.TSSetIJacobianP

        """
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        if jacobian is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (jacobian, args, kargs)
            self.set_attr('__ijacobianp__', context)
            CHKERR(TSSetIJacobianP(self.ts, Jmat, TS_IJacobianP, <void*>context))
        else:
            CHKERR(TSSetIJacobianP(self.ts, Jmat, NULL, NULL))

    def computeIFunction(self,
                         t: float, Vec x, Vec xdot,
                         Vec f, imex: bool = False) -> None:
        """Evaluate the DAE residual written in implicit form.

        Collective.

        Parameters
        ----------
        t
            The current time.
        x
            The state vector.
        xdot
            The time derivative of the state vector.
        f
            The vector into which the residual is stored.
        imex
            A flag which indicates if the RHS should be kept separate.

        See Also
        --------
        petsc.TSComputeIFunction

        """
        cdef PetscReal rval = asReal(t)
        cdef PetscBool bval = imex
        CHKERR(TSComputeIFunction(self.ts, rval, x.vec, xdot.vec,
                                  f.vec, bval))

    def computeIJacobian(self,
                         t: float, Vec x, Vec xdot, a: float,
                         Mat J, Mat P = None, imex: bool = False) -> None:
        """Evaluate the Jacobian of the DAE.

        Collective.

        If ``F(t, U, Udot)=0`` is the DAE, the required Jacobian is
        ``dF/dU + shift*dF/dUdot``

        Parameters
        ----------
        t
            The current time.
        x
            The state vector.
        xdot
            The time derivative of the state vector.
        a
            The shift to apply
        J
            The matrix into which the Jacobian is computed.
        P
            The optional matrix to use for building a preconditioner.
        imex
            A flag which indicates if the RHS should be kept separate.

        See Also
        --------
        petsc.TSComputeIJacobian

        """
        cdef PetscReal rval1 = asReal(t)
        cdef PetscReal rval2 = asReal(a)
        cdef PetscBool bval  = imex
        cdef PetscMat jmat = J.mat, pmat = J.mat
        if P is not None: pmat = P.mat
        CHKERR(TSComputeIJacobian(self.ts, rval1, x.vec, xdot.vec, rval2,
                                  jmat, pmat, bval))

    def computeIJacobianP(self,
                          t: float, Vec x, Vec xdot, a: float,
                          Mat J, imex: bool = False) -> None:
        """Evaluate the Jacobian with respect to parameters.

        Collective.

        Parameters
        ----------
        t
            The current time.
        x
            The state vector.
        xdot
            The time derivative of the state vector.
        a
            The shift to apply
        J
            The matrix into which the Jacobian is computed.
        imex
            A flag which indicates if the RHS should be kept separate.

        See Also
        --------
        petsc.TSComputeIJacobianP

        """
        cdef PetscReal rval1 = asReal(t)
        cdef PetscReal rval2 = asReal(a)
        cdef PetscBool bval  = asBool(imex)
        cdef PetscMat jmat = J.mat
        CHKERR(TSComputeIJacobianP(self.ts, rval1, x.vec, xdot.vec, rval2,
                                   jmat, bval))

    def getIFunction(self) -> tuple[Vec, TSIFunction]:
        """Return the vector and function which computes the implicit residual.

        Not collective.

        See Also
        --------
        petsc.TSGetIFunction

        """
        cdef Vec f = Vec()
        CHKERR(TSGetIFunction(self.ts, &f.vec, NULL, NULL))
        CHKERR(PetscINCREF(f.obj))
        cdef object function = self.get_attr('__ifunction__')
        return (f, function)

    def getIJacobian(self) -> tuple[Mat, Mat, TSIJacobian]:
        """Return the matrices and function which computes the implicit Jacobian.

        Not collective.

        See Also
        --------
        petsc.TSGetIJacobian

        """
        cdef Mat J = Mat(), P = Mat()
        CHKERR(TSGetIJacobian(self.ts, &J.mat, &P.mat, NULL, NULL))
        CHKERR(PetscINCREF(J.obj)); CHKERR(PetscINCREF(P.obj))
        cdef object jacobian = self.get_attr('__ijacobian__')
        return (J, P, jacobian)

    def setI2Function(
        self,
        function: TSI2Function | None,
        Vec f=None,
        args : tuple[Any, ...] | None = None,
        kargs : dict[str, Any] | None = None) -> None:
        """Set the function to compute the 2nd order DAE.

        Logically collective.

        Parameters
        ----------
        function
            The right-hand side function.
        f
            The vector to store values or `None` to be created internally.
        args
            Additional positional arguments for ``function``.
        kargs
            Additional keyword arguments for ``function``.

        See Also
        --------
        petsc.TSSetI2Function

        """
        cdef PetscVec fvec=NULL
        if f is not None: fvec = f.vec
        if function is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (function, args, kargs)
            self.set_attr('__i2function__', context)
            CHKERR(TSSetI2Function(self.ts, fvec, TS_I2Function, <void*>context))
        else:
            CHKERR(TSSetI2Function(self.ts, fvec, NULL, NULL))

    def setI2Jacobian(
        self,
        jacobian: TSI2Jacobian | None,
        Mat J=None,
        Mat P=None,
        args=None,
        kargs=None) -> None:
        """Set the function to compute the Jacobian of the 2nd order DAE.

        Logically collective.

        Parameters
        ----------
        jacobian
            The function which computes the Jacobian.
        J
            The matrix into which the Jacobian is computed.
        P
            The optional matrix to use for building a preconditioner.
        args
            Additional positional arguments for ``jacobian``.
        kargs
            Additional keyword arguments for ``jacobian``.

        See Also
        --------
        petsc.TSSetI2Jacobian

        """
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat=Jmat
        if P is not None: Pmat = P.mat
        if jacobian is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (jacobian, args, kargs)
            self.set_attr('__i2jacobian__', context)
            CHKERR(TSSetI2Jacobian(self.ts, Jmat, Pmat, TS_I2Jacobian, <void*>context))
        else:
            CHKERR(TSSetI2Jacobian(self.ts, Jmat, Pmat, NULL, NULL))

    def computeI2Function(self, t: float, Vec x, Vec xdot, Vec xdotdot, Vec f) -> None:
        """Evaluate the DAE residual in implicit form.

        Collective.

        Parameters
        ----------
        t
            The current time.
        x
            The state vector.
        xdot
            The time derivative of the state vector.
        xdotdot
            The second time derivative of the state vector.
        f
            The vector into which the residual is stored.

        See Also
        --------
        petsc.TSComputeI2Function

        """
        cdef PetscReal rval = asReal(t)
        CHKERR(TSComputeI2Function(self.ts, rval, x.vec, xdot.vec, xdotdot.vec,
                                   f.vec))

    def computeI2Jacobian(
        self,
        t: float,
        Vec x,
        Vec xdot,
        Vec xdotdot,
        v: float,
        a: float,
        Mat J,
        Mat P=None) -> None:
        """Evaluate the Jacobian of the DAE.

        Collective.

        If ``F(t, U, V, A)=0`` is the DAE,
        the required Jacobian is ``dF/dU + v dF/dV + a dF/dA``.

        Parameters
        ----------
        t
            The current time.
        x
            The state vector.
        xdot
            The time derivative of the state vector.
        xdotdot
            The second time derivative of the state vector.
        v
            The shift to apply to the first derivative.
        a
            The shift to apply to the second derivative.
        J
            The matrix into which the Jacobian is computed.
        P
            The optional matrix to use for building a preconditioner.

        See Also
        --------
        petsc.TSComputeI2Jacobian

        """
        cdef PetscReal rval1 = asReal(t)
        cdef PetscReal rval2 = asReal(v)
        cdef PetscReal rval3 = asReal(a)
        cdef PetscMat jmat = J.mat, pmat = J.mat
        if P is not None: pmat = P.mat
        CHKERR(TSComputeI2Jacobian(self.ts, rval1, x.vec, xdot.vec, xdotdot.vec, rval2, rval3,
                                   jmat, pmat))

    def getI2Function(self) -> tuple[Vec, TSI2Function]:
        """Return the vector and function which computes the residual.

        Not collective.

        See Also
        --------
        petsc.TSGetI2Function

        """
        cdef Vec f = Vec()
        CHKERR(TSGetI2Function(self.ts, &f.vec, NULL, NULL))
        CHKERR(PetscINCREF(f.obj))
        cdef object function = self.get_attr('__i2function__')
        return (f, function)

    def getI2Jacobian(self) -> tuple[Mat, Mat, TSI2Jacobian]:
        """Return the matrices and function which computes the Jacobian.

        Not collective.

        See Also
        --------
        petsc.TSGetI2Jacobian

        """
        cdef Mat J = Mat(), P = Mat()
        CHKERR(TSGetI2Jacobian(self.ts, &J.mat, &P.mat, NULL, NULL))
        CHKERR(PetscINCREF(J.obj)); CHKERR(PetscINCREF(P.obj))
        cdef object jacobian = self.get_attr('__i2jacobian__')
        return (J, P, jacobian)

    # --- TSRHSSplit routines to support multirate and IMEX solvers ---
    def setRHSSplitIS(self, splitname: str, IS iss) -> None:
        """Set the index set for the specified split.

        Logically collective.

        Parameters
        ----------
        splitname
            Name of this split, if `None` the number of the split is used.
        iss
            The index set for part of the solution vector.

        See Also
        --------
        petsc.TSRHSSplitSetIS

        """
        cdef const char *cname = NULL
        splitname = str2bytes(splitname, &cname)
        CHKERR(TSRHSSplitSetIS(self.ts, cname, iss.iset))

    def setRHSSplitRHSFunction(
        self,
        splitname: str,
        function: TSRHSFunction,
        Vec r=None,
        args : tuple[Any, ...] | None = None,
        kargs : dict[str, Any] | None = None) -> None:
        """Set the split right-hand-side functions.

        Logically collective.

        Parameters
        ----------
        splitname
            Name of the split.
        function
            The RHS function evaluation routine.
        r
            Vector to hold the residual.
        args
            Additional positional arguments for ``function``.
        kargs
            Additional keyword arguments for ``function``.

        See Also
        -------
        petsc.TSRHSSplitSetRHSFunction

        """
        cdef const char *cname = NULL
        cdef char *aname = NULL
        splitname = str2bytes(splitname, <const char**>&cname)
        cdef PetscVec rvec=NULL
        if r is not None: rvec = r.vec
        if function is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (function, args, kargs)
            str2bytes(function.__name__, <const char**>&aname)
            self.set_attr(aname, context) # to avoid being GCed
            CHKERR(TSRHSSplitSetRHSFunction(self.ts, cname, rvec, TS_RHSFunction, <void*>context))
        else:
            CHKERR(TSRHSSplitSetRHSFunction(self.ts, cname, rvec, NULL, NULL))

    def setRHSSplitIFunction(
        self,
        splitname: str,
        function: TSIFunction,
        Vec r=None,
        args : tuple[Any, ...] | None = None,
        kargs : dict[str, Any] | None = None) -> None:
        """Set the split implicit functions.

        Logically collective.

        Parameters
        ----------
        splitname
            Name of the split.
        function
            The implicit function evaluation routine.
        r
            Vector to hold the residual.
        args
            Additional positional arguments for ``function``.
        kargs
            Additional keyword arguments for ``function``.

        See Also
        -------
        petsc.TSRHSSplitSetIFunction

        """
        cdef const char *cname = NULL
        cdef char *aname = NULL
        splitname = str2bytes(splitname, &cname)
        cdef PetscVec rvec=NULL
        if r is not None: rvec = r.vec
        if function is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (function, args, kargs)
            str2bytes(function.__name__, <const char**>&aname)
            self.set_attr(aname, context) # to avoid being GCed
            CHKERR(TSRHSSplitSetIFunction(self.ts, cname, rvec, TS_IFunction, <void*>context))
        else:
            CHKERR(TSRHSSplitSetIFunction(self.ts, cname, rvec, NULL, NULL))

    def setRHSSplitIJacobian(
        self,
        splitname: str,
        jacobian: TSRHSJacobian,
        Mat J=None,
        Mat P=None,
        args : tuple[Any, ...] | None = None,
        kargs : dict[str, Any] | None = None) -> None:
        """Set the Jacobian for the split implicit function.

        Logically collective.

        Parameters
        ----------
        splitname
            Name of the split.
        jacobian
            The Jacobian evaluation routine.
        J
            Matrix to store Jacobian entries computed by ``jacobian``.
        P
            Matrix used to compute preconditioner (usually the same as ``J``).
        args
            Additional positional arguments for ``jacobian``.
        kargs
            Additional keyword arguments for ``jacobian``.

        See Also
        -------
        petsc.TSRHSSplitSetIJacobian

        """
        cdef const char *cname = NULL
        cdef char *aname = NULL
        splitname = str2bytes(splitname, &cname)
        cdef PetscMat Jmat=NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat=Jmat
        if P is not None: Pmat = P.mat
        if jacobian is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (jacobian, args, kargs)
            str2bytes(jacobian.__name__, <const char**>&aname)
            self.set_attr(aname, context) # to avoid being GCed
            CHKERR(TSRHSSplitSetIJacobian(self.ts, cname, Jmat, Pmat, TS_IJacobian, <void*>context))
        else:
            CHKERR(TSRHSSplitSetIJacobian(self.ts, cname, Jmat, Pmat, NULL, NULL))

    # --- solution vector ---

    def setSolution(self, Vec u) -> None:
        """Set the initial solution vector.

        Logically collective.

        Parameters
        ----------
        u
            The solution vector.

        See Also
        --------
        petsc.TSSetSolution

        """
        CHKERR(TSSetSolution(self.ts, u.vec))

    def getSolution(self) -> Vec:
        """Return the solution at the present timestep.

        Not collective.

        It is valid to call this routine inside the function that you are
        evaluating in order to move to the new timestep. This vector is not
        changed until the solution at the next timestep has been calculated.

        See Also
        --------
        petsc.TSGetSolution

        """
        cdef Vec u = Vec()
        CHKERR(TSGetSolution(self.ts, &u.vec))
        CHKERR(PetscINCREF(u.obj))
        return u

    def setSolution2(self, Vec u, Vec v) -> None:
        """Set the initial solution and its time derivative.

        Logically collective.

        Parameters
        ----------
        u
            The solution vector.
        v
            The time derivative vector.

        See Also
        --------
        petsc.TS2SetSolution

        """
        CHKERR(TS2SetSolution(self.ts, u.vec, v.vec))

    def getSolution2(self) -> tuple[Vec, Vec]:
        """Return the solution and time derivative at the present timestep.

        Not collective.

        It is valid to call this routine inside the function that you are
        evaluating in order to move to the new timestep. These vectors are not
        changed until the solution at the next timestep has been calculated.

        See Also
        --------
        petsc.TS2GetSolution

        """
        cdef Vec u = Vec()
        cdef Vec v = Vec()
        CHKERR(TS2GetSolution(self.ts, &u.vec, &v.vec))
        CHKERR(PetscINCREF(u.obj))
        CHKERR(PetscINCREF(v.obj))
        return (u, v)

    # --- evaluation times ---

    def setEvaluationTimes(self, tspan: Sequence[float]) -> None:
        """Sets evaluation points where solution will be computed and stored.

        Collective.

        The solution will be computed and stored for each time
        requested. The times must be all increasing and correspond
        to the intermediate points for time integration.
        `ExactFinalTime.MATCHSTEP` must be used to make the last time step in
        each sub-interval match the intermediate points specified. The
        intermediate solutions are saved in a vector array that can be accessed
        with `getEvaluationSolutions`.

        Parameters
        ----------
        tspan
            The sequence of time points. The first element and the last element
            are the initial time and the final time respectively.

        Notes
        -----
        ``-ts_eval_times <t0, ..., tn>`` sets the time span from the commandline

        See Also
        --------
        getEvaluationTimes, petsc.TSGetEvaluationTimes

        """
        cdef PetscInt  nt = 0
        cdef PetscReal *rtspan = NULL
        cdef unused = oarray_r(tspan, &nt, &rtspan)
        CHKERR(TSSetEvaluationTimes(self.ts, nt, rtspan))

    def getEvaluationTimes(self) -> ArrayReal:
        """Return the evaluation points.

        Not collective.

        See Also
        --------
        setEvaluationTimes

        """
        cdef const PetscReal *rtspan = NULL
        cdef PetscInt   nt = 0
        CHKERR(TSGetEvaluationTimes(self.ts, &nt, &rtspan))
        cdef object tspan = array_r(nt, rtspan)
        return tspan

    def getEvaluationSolutions(self) -> tuple[ArrayReal, list[Vec]]:
        """Return the solutions and the times they were recorded at.

        Not collective.

        See Also
        --------
        setEvaluationTimes

        """
        cdef PetscInt nt = 0
        cdef PetscVec *sols = NULL
        cdef const PetscReal *rtspan = NULL
        CHKERR(TSGetEvaluationSolutions(self.ts, &nt, &rtspan, &sols))
        cdef object sollist = None
        if sols != NULL:
            sollist = [ref_Vec(sols[i]) for i from 0 <= i < nt]
        cdef object tspan = array_r(nt, rtspan)
        return tspan, sollist

    # --- time span ---

    def setTimeSpan(self, tspan: Sequence[float]) -> None:
        """Set the time span and time points to evaluate solution at.

        Collective.

        The solution will be computed and stored for each time
        requested in the span. The times must be all increasing and correspond
        to the intermediate points for time integration.
        `ExactFinalTime.MATCHSTEP` must be used to make the last time step in
        each sub-interval match the intermediate points specified. The
        intermediate solutions are saved in a vector array that can be accessed
        with `getEvaluationSolutions`.

        Parameters
        ----------
        tspan
            The sequence of time points. The first element and the last element
            are the initial time and the final time respectively.

        Notes
        -----
        ``-ts_time_span <t0, ..., tf>`` sets the time span from the commandline

        See Also
        --------
        setEvaluationTimes, petsc.TSSetTimeSpan

        """
        cdef PetscInt  nt = 0
        cdef PetscReal *rtspan = NULL
        cdef unused = oarray_r(tspan, &nt, &rtspan)
        CHKERR(TSSetTimeSpan(self.ts, nt, rtspan))

    getTimeSpan = getEvaluationTimes

    def getTimeSpanSolutions(self) -> list[Vec]:
        """Return the solutions at the times in the time span. Deprecated.

        Not collective.

        See Also
        --------
        setTimeSpan, setEvaluationTimes, getEvaluationSolutions

        """
        cdef PetscInt nt = 0
        cdef PetscVec *sols = NULL
        CHKERR(TSGetEvaluationSolutions(self.ts, &nt, NULL, &sols))
        cdef object sollist = None
        if sols != NULL:
            sollist = [ref_Vec(sols[i]) for i from 0 <= i < nt]
        return sollist

    # --- inner solver ---

    def getSNES(self) -> SNES:
        """Return the `SNES` associated with the `TS`.

        Not collective.

        See Also
        --------
        petsc.TSGetSNES

        """
        cdef SNES snes = SNES()
        CHKERR(TSGetSNES(self.ts, &snes.snes))
        CHKERR(PetscINCREF(snes.obj))
        return snes

    def getKSP(self) -> KSP:
        """Return the `KSP` associated with the `TS`.

        Not collective.

        See Also
        --------
        petsc.TSGetKSP

        """
        cdef KSP ksp = KSP()
        CHKERR(TSGetKSP(self.ts, &ksp.ksp))
        CHKERR(PetscINCREF(ksp.obj))
        return ksp

    # --- discretization space ---

    def getDM(self) -> DM:
        """Return the `DM` associated with the `TS`.

        Not collective.

        Only valid if nonlinear solvers or preconditioners are
        used which use the `DM`.

        See Also
        --------
        petsc.TSGetDM

        """
        cdef PetscDM newdm = NULL
        CHKERR(TSGetDM(self.ts, &newdm))
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        CHKERR(PetscINCREF(dm.obj))
        return dm

    def setDM(self, DM dm) -> None:
        """Set the DM that may be used by some nonlinear solvers or preconditioners.

        Logically collective.

        Parameters
        ----------
        dm
            The `DM` object.

        See Also
        --------
        petsc.TSSetDM

        """
        CHKERR(TSSetDM(self.ts, dm.dm))

    # --- customization ---

    def setTime(self, t: float) -> None:
        """Set the time.

        Logically collective.

        Parameters
        ----------
        t
            The time.

        See Also
        --------
        petsc.TSSetTime

        """
        cdef PetscReal rval = asReal(t)
        CHKERR(TSSetTime(self.ts, rval))

    def getTime(self) -> float:
        """Return the time of the most recently completed step.

        Not collective.

        When called during time step evaluation (e.g. during
        residual evaluation or via hooks set using `setPreStep` or
        `setPostStep`), the time returned is at the start of the step.

        See Also
        --------
        petsc.TSGetTime

        """
        cdef PetscReal rval = 0
        CHKERR(TSGetTime(self.ts, &rval))
        return toReal(rval)

    def getPrevTime(self) -> float:
        """Return the starting time of the previously completed step.

        Not collective.

        See Also
        --------
        petsc.TSGetPrevTime

        """
        cdef PetscReal rval = 0
        CHKERR(TSGetPrevTime(self.ts, &rval))
        return toReal(rval)

    def getSolveTime(self) -> float:
        """Return the time after a call to `solve`.

        Not collective.

        This time corresponds to the final time set with
        `setMaxTime`.

        See Also
        --------
        petsc.TSGetSolveTime

        """
        cdef PetscReal rval = 0
        CHKERR(TSGetSolveTime(self.ts, &rval))
        return toReal(rval)

    def setTimeStep(self, time_step: float) -> None:
        """Set the duration of the timestep.

        Logically collective.

        Parameters
        ----------
        time_step
            the duration of the timestep

        See Also
        --------
        petsc.TSSetTimeStep

        """
        cdef PetscReal rval = asReal(time_step)
        CHKERR(TSSetTimeStep(self.ts, rval))

    def getTimeStep(self) -> float:
        """Return the duration of the current timestep.

        Not collective.

        See Also
        --------
        petsc.TSGetTimeStep

        """
        cdef PetscReal tstep = 0
        CHKERR(TSGetTimeStep(self.ts, &tstep))
        return toReal(tstep)

    def setStepNumber(self, step_number: int) -> None:
        """Set the number of steps completed.

        Logically collective.

        For most uses of the `TS` solvers the user need
        not explicitly call `setStepNumber`, as the step counter is
        appropriately updated in `solve`/`step`/`rollBack`. Power users may call
        this routine to reinitialize timestepping by setting the step counter to
        zero (and time to the initial time) to solve a similar problem with
        different initial conditions or parameters. It may also be used to
        continue timestepping from a previously interrupted run in such a way
        that `TS` monitors will be called with a initial nonzero step counter.

        Parameters
        ----------
        step_number
            the number of steps completed

        See Also
        --------
        petsc.TSSetStepNumber

        """
        cdef PetscInt ival = asInt(step_number)
        CHKERR(TSSetStepNumber(self.ts, ival))

    def getStepNumber(self) -> int:
        """Return the number of time steps completed.

        Not collective.

        See Also
        --------
        petsc.TSGetStepNumber

        """
        cdef PetscInt ival = 0
        CHKERR(TSGetStepNumber(self.ts, &ival))
        return toInt(ival)

    def setMaxTime(self, max_time: float) -> None:
        """Set the maximum (final) time.

        Logically collective.

        Parameters
        ----------
        max_time
            the final time

        Notes
        -----
        ``-ts_max_time`` sets the max time from the commandline

        See Also
        --------
        petsc.TSSetMaxTime

        """
        cdef PetscReal rval = asReal(max_time)
        CHKERR(TSSetMaxTime(self.ts, rval))

    def getMaxTime(self) -> float:
        """Return the maximum (final) time.

        Not collective.

        Defaults to ``5``.

        See Also
        --------
        petsc.TSGetMaxTime

        """
        cdef PetscReal rval = 0
        CHKERR(TSGetMaxTime(self.ts, &rval))
        return toReal(rval)

    def setMaxSteps(self, max_steps: int) -> None:
        """Set the maximum number of steps to use.

        Logically collective.

        Defaults to ``5000``.

        Parameters
        ----------
        max_steps
            The maximum number of steps to use.

        See Also
        --------
        petsc.TSSetMaxSteps

        """
        cdef PetscInt  ival = asInt(max_steps)
        CHKERR(TSSetMaxSteps(self.ts, ival))

    def getMaxSteps(self) -> int:
        """Return the maximum number of steps to use.

        Not collective.

        See Also
        --------
        petsc.TSGetMaxSteps

        """
        cdef PetscInt ival = 0
        CHKERR(TSGetMaxSteps(self.ts, &ival))
        return toInt(ival)

    def getSNESIterations(self) -> int:
        """Return the total number of nonlinear iterations used by the `TS`.

        Not collective.

        This counter is reset to zero for each successive call
        to `solve`.

        See Also
        --------
        petsc.TSGetSNESIterations

        """
        cdef PetscInt n = 0
        CHKERR(TSGetSNESIterations(self.ts, &n))
        return toInt(n)

    def getKSPIterations(self) -> int:
        """Return the total number of linear iterations used by the `TS`.

        Not collective.

        This counter is reset to zero for each successive call
        to `solve`.

        See Also
        --------
        petsc.TSGetKSPIterations

        """
        cdef PetscInt n = 0
        CHKERR(TSGetKSPIterations(self.ts, &n))
        return toInt(n)

    def setMaxStepRejections(self, n: int) -> None:
        """Set the maximum number of step rejections before a time step fails.

        Not collective.

        Parameters
        ----------
        n
            The maximum number of rejected steps, use ``-1`` for unlimited.

        Notes
        -----
        ``-ts_max_reject`` can be used to set this from the commandline

        See Also
        --------
        petsc.TSSetMaxStepRejections

        """
        cdef PetscInt rej = asInt(n)
        CHKERR(TSSetMaxStepRejections(self.ts, rej))

    def getStepRejections(self) -> int:
        """Return the total number of rejected steps.

        Not collective.

        This counter is reset to zero for each successive call
        to `solve`.

        See Also
        --------
        petsc.TSGetStepRejections

        """
        cdef PetscInt n = 0
        CHKERR(TSGetStepRejections(self.ts, &n))
        return toInt(n)

    def setMaxSNESFailures(self, n: int) -> None:
        """Set the maximum number of SNES solves failures allowed.

        Not collective.

        Parameters
        ----------
        n
            The maximum number of failed nonlinear solver, use ``-1`` for unlimited.

        See Also
        --------
        petsc.TSSetMaxSNESFailures

        """
        cdef PetscInt fails = asInt(n)
        CHKERR(TSSetMaxSNESFailures(self.ts, fails))

    def getSNESFailures(self) -> int:
        """Return the total number of failed `SNES` solves in the `TS`.

        Not collective.

        This counter is reset to zero for each successive call
        to `solve`.

        See Also
        --------
        petsc.TSGetSNESFailures

        """
        cdef PetscInt n = 0
        CHKERR(TSGetSNESFailures(self.ts, &n))
        return toInt(n)

    def setErrorIfStepFails(self, flag: bool = True) -> None:
        """Immediately error is no step succeeds.

        Not collective.

        Parameters
        ----------
        flag
            Enable to error if no step succeeds.

        Notes
        -----
        ``-ts_error_if_step_fails`` to enable from the commandline.

        See Also
        --------
        petsc.TSSetErrorIfStepFails

        """
        cdef PetscBool bval = flag
        CHKERR(TSSetErrorIfStepFails(self.ts, bval))

    def setTolerances(self, rtol: float | None = None, atol: float | None = None) -> None:
        """Set tolerances for local truncation error when using an adaptive controller.

        Logically collective.

        Parameters
        ----------
        rtol
            The relative tolerance, `DETERMINE` to use the value
            when the object's type was set, or `None` to leave the
            current value.
        atol
            The absolute tolerance, `DETERMINE` to use the
            value when the object's type was set, or `None` to
            leave the current value.

        Notes
        -----
        ``-ts_rtol`` and ``-ts_atol`` may be used to set values from the commandline.

        See Also
        --------
        petsc.TSSetTolerances

        """
        cdef PetscReal rrtol = PETSC_CURRENT
        cdef PetscReal ratol = PETSC_CURRENT
        cdef PetscVec  vrtol = NULL
        cdef PetscVec  vatol = NULL
        if rtol is None:
            pass
        elif isinstance(rtol, Vec):
            vrtol = (<Vec>rtol).vec
        else:
            rrtol = asReal(rtol)
        if atol is None:
            pass
        elif isinstance(atol, Vec):
            vatol = (<Vec>atol).vec
        else:
            ratol = asReal(atol)
        CHKERR(TSSetTolerances(self.ts, ratol, vatol, rrtol, vrtol))

    def getTolerances(self) ->tuple[float, float]:
        """Return the tolerances for local truncation error.

        Logically collective.

        Returns
        -------
        rtol : float
            the relative tolerance
        atol : float
            the absolute tolerance

        See Also
        --------
        petsc.TSGetTolerances

        """
        cdef PetscReal rrtol = PETSC_DETERMINE
        cdef PetscReal ratol = PETSC_DETERMINE
        cdef PetscVec  vrtol = NULL
        cdef PetscVec  vatol = NULL
        CHKERR(TSGetTolerances(self.ts, &ratol, &vatol, &rrtol, &vrtol))
        cdef object rtol = None
        if vrtol != NULL:
            rtol = ref_Vec(vrtol)
        else:
            rtol = toReal(rrtol)
        cdef object atol = None
        if vatol != NULL:
            atol = ref_Vec(vatol)
        else:
            atol = toReal(ratol)
        return (rtol, atol)

    def setExactFinalTime(self, option: ExactFinalTime) -> None:
        """Set method of computing the final time step.

        Logically collective.

        Parameters
        ----------
        option
            The exact final time option

        Notes
        -----
        ``-ts_exact_final_time`` may be used to specify from the commandline.

        See Also
        --------
        petsc.TSSetExactFinalTime

        """
        cdef PetscTSExactFinalTimeOption oval = option
        CHKERR(TSSetExactFinalTime(self.ts, oval))

    def setConvergedReason(self, reason: ConvergedReason) -> None:
        """Set the reason for handling the convergence of `solve`.

        Logically collective.

        Can only be called when `solve` is active and
        ``reason`` must contain common value.

        Parameters
        ----------
        reason
            The reason for convergence.

        See Also
        --------
        petsc.TSSetConvergedReason

        """
        cdef PetscTSConvergedReason cval = reason
        CHKERR(TSSetConvergedReason(self.ts, cval))

    def getConvergedReason(self) -> ConvergedReason:
        """Return the reason the `TS` step was stopped.

        Not collective.

        Can only be called once `solve` is complete.

        See Also
        --------
        petsc.TSGetConvergedReason

        """
        cdef PetscTSConvergedReason reason = TS_CONVERGED_ITERATING
        CHKERR(TSGetConvergedReason(self.ts, &reason))
        return reason

    # --- monitoring ---

    def setMonitor(
        self,
        monitor: TSMonitorFunction | None,
        args : tuple[Any, ...] | None = None,
        kargs : dict[str, Any] | None = None) -> None:
        """Set an additional monitor to the `TS`.

        Logically collective.

        Parameters
        ----------
        monitor
            The custom monitor function.
        args
            Additional positional arguments for ``monitor``.
        kargs
            Additional keyword arguments for ``monitor``.

        See Also
        --------
        petsc.TSMonitorSet

        """
        if monitor is None: return
        cdef object monitorlist = self.get_attr('__monitor__')
        if monitorlist is None:
            monitorlist = []
            self.set_attr('__monitor__', monitorlist)
            CHKERR(TSMonitorSet(self.ts, TS_Monitor, NULL, NULL))
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (monitor, args, kargs)
        monitorlist.append(context)

    def getMonitor(self) -> list[tuple[TSMonitorFunction, tuple[Any, ...], dict[str, Any]]]:
        """Return the monitor.

        Not collective.

        See Also
        --------
        setMonitor

        """
        return self.get_attr('__monitor__')

    def monitorCancel(self) -> None:
        """Clear all the monitors that have been set.

        Logically collective.

        See Also
        --------
        petsc.TSMonitorCancel

        """
        self.set_attr('__monitor__', None)
        CHKERR(TSMonitorCancel(self.ts))

    cancelMonitor = monitorCancel

    def monitor(self, step: int, time: float, Vec u=None) -> None:
        """Monitor the solve.

        Collective.

        Parameters
        ----------
        step
            The step number that has just completed.
        time
            The model time of the state.
        u
            The state at the current model time.

        See Also
        --------
        petsc.TSMonitor

        """
        cdef PetscInt  ival = asInt(step)
        cdef PetscReal rval = asReal(time)
        cdef PetscVec  uvec = NULL
        if u is not None: uvec = u.vec
        if uvec == NULL:
            CHKERR(TSGetSolution(self.ts, &uvec))
        CHKERR(TSMonitor(self.ts, ival, rval, uvec))

    # --- event handling ---

    def setEventHandler(
        self,
        direction: Sequence[int],
        terminate: Sequence[bool],
        indicator: TSIndicatorFunction | None,
        postevent: TSPostEventFunction | None = None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set a function used for detecting events.

        Logically collective.

        Parameters
        ----------
        direction
            Direction of zero crossing to be detected {-1, 0, +1}.
        terminate
            Flags for each event to indicate stepping should be terminated.
        indicator
            Function for defining the indicator-functions marking the events
        postevent
            Function to execute after the event
        args
            Additional positional arguments for ``indicator``.
        kargs
            Additional keyword arguments for ``indicator``.

        See Also
        --------
        petsc.TSSetEventHandler

        """
        cdef PetscInt  ndirs = 0
        cdef PetscInt *idirs = NULL
        direction = iarray_i(direction, &ndirs, &idirs)

        cdef PetscInt   nterm = 0
        cdef PetscBool *iterm = NULL
        terminate = iarray_b(terminate, &nterm, &iterm)
        assert nterm == ndirs

        cdef PetscInt nevents = ndirs
        if indicator is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            self.set_attr('__indicator__', (indicator, args, kargs))
            if postevent is not None:
                self.set_attr('__postevent__', (postevent, args, kargs))
                CHKERR(TSSetEventHandler(self.ts, nevents, idirs, iterm, TS_Indicator, TS_PostEvent, <void*>NULL))
            else:
                self.set_attr('__postevent__', None)
                CHKERR(TSSetEventHandler(self.ts, nevents, idirs, iterm, TS_Indicator, NULL, <void*>NULL))
        else:
            CHKERR(TSSetEventHandler(self.ts, nevents, idirs, iterm, NULL, NULL, <void*>NULL))

    def setEventTolerances(self, tol: float | None = None, vtol: Sequence[float] | None = None) -> None:
        """Set tolerances for event zero crossings when using event handler.

        Logically collective.

        ``setEventHandler`` must have already been called.

        Parameters
        ----------
        tol
            The scalar tolerance or `None` to leave at the current value
        vtol
            A sequence of scalar tolerance for each event. Used in preference to
            ``tol`` if present. Set to `None` to leave at the current value.

        Notes
        -----
        ``-ts_event_tol`` can be used to set values from the commandline.

        See Also
        --------
        petsc.TSSetEventTolerances

        """
        cdef PetscInt  nevents = 0
        cdef PetscReal tolr = PETSC_CURRENT
        cdef PetscInt  ntolr = 0
        cdef PetscReal *vtolr = NULL
        if tol is not None:
            tolr = asReal(tol)
        if vtol is not None:
            CHKERR(TSGetNumEvents(self.ts, &nevents))
            vtol = iarray_r(vtol, &ntolr,  &vtolr)
            assert ntolr == nevents
        CHKERR(TSSetEventTolerances(self.ts, tolr, vtolr))

    def getNumEvents(self) -> int:
        """Return the number of events.

        Logically collective.

        See Also
        --------
        petsc.TSGetNumEvents

        """
        cdef PetscInt nevents = 0
        CHKERR(TSGetNumEvents(self.ts, &nevents))
        return toInt(nevents)

    # --- solving ---

    def setPreStep(
        self,
        prestep: TSPreStepFunction | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set a function to be called at the beginning of each time step.

        Logically collective.

        Parameters
        ----------
        prestep
            The function to be called at the beginning of each step.
        args
            Additional positional arguments for ``prestep``.
        kargs
            Additional keyword arguments for ``prestep``.

        See Also
        --------
        petsc.TSSetPreStep

        """
        if prestep is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (prestep, args, kargs)
            self.set_attr('__prestep__', context)
            CHKERR(TSSetPreStep(self.ts, TS_PreStep))
        else:
            self.set_attr('__prestep__', None)
            CHKERR(TSSetPreStep(self.ts, NULL))

    def getPreStep(self) -> tuple[TSPreStepFunction, tuple[Any, ...] | None, dict[str, Any] | None]:
        """Return the prestep function.

        Not collective.

        See Also
        --------
        setPreStep

        """
        return self.get_attr('__prestep__')

    def setPostStep(self,
                    poststep: TSPostStepFunction | None,
                    args: tuple[Any, ...] | None = None,
                    kargs: dict[str, Any] | None = None) -> None:
        """Set a function to be called at the end of each time step.

        Logically collective.

        Parameters
        ----------
        poststep
            The function to be called at the end of each step.
        args
            Additional positional arguments for ``poststep``.
        kargs
            Additional keyword arguments for ``poststep``.

        See Also
        --------
        petsc.TSSetPostStep

        """
        if poststep is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (poststep, args, kargs)
            self.set_attr('__poststep__', context)
            CHKERR(TSSetPostStep(self.ts, TS_PostStep))
        else:
            self.set_attr('__poststep__', None)
            CHKERR(TSSetPostStep(self.ts, NULL))

    def getPostStep(self) -> tuple[TSPostStepFunction, tuple[Any, ...] | None, dict[str, Any] | None]:
        """Return the poststep function."""
        return self.get_attr('__poststep__')

    def setUp(self) -> None:
        """Set up the internal data structures for the `TS`.

        Collective.

        See Also
        --------
        petsc.TSSetUp

        """
        CHKERR(TSSetUp(self.ts))

    def reset(self) -> None:
        """Reset the `TS`, removing any allocated vectors and matrices.

        Collective.

        See Also
        --------
        petsc.TSReset

        """
        CHKERR(TSReset(self.ts))

    def step(self) -> None:
        """Take one step.

        Collective.

        The preferred interface for the `TS` solvers is `solve`. If
        you need to execute code at the beginning or ending of each step, use
        `setPreStep` and `setPostStep` respectively.

        See Also
        --------
        petsc.TSStep

        """
        CHKERR(TSStep(self.ts))

    def restartStep(self) -> None:
        """Flag the solver to restart the next step.

        Collective.

        Multistep methods like TSBDF or Runge-Kutta methods with
        FSAL property require restarting the solver in the event of
        discontinuities. These discontinuities may be introduced as a
        consequence of explicitly modifications to the solution vector (which
        PETSc attempts to detect and handle) or problem coefficients (which
        PETSc is not able to detect). For the sake of correctness and maximum
        safety, users are expected to call TSRestart() whenever they introduce
        discontinuities in callback routines (e.g. prestep and poststep
        routines, or implicit/rhs function routines with discontinuous source
        terms).

        See Also
        --------
        petsc.TSRestartStep

        """
        CHKERR(TSRestartStep(self.ts))

    def rollBack(self) -> None:
        """Roll back one time step.

        Collective.

        See Also
        --------
        petsc.TSRollBack

        """
        CHKERR(TSRollBack(self.ts))

    def solve(self, Vec u=None) -> None:
        """Step the requested number of timesteps.

        Collective.

        Parameters
        ----------
        u
            The solution vector. Can be `None`.

        See Also
        --------
        petsc.TSSolve

        """
        cdef PetscVec uvec=NULL
        if u is not None: uvec = u.vec
        CHKERR(TSSolve(self.ts, uvec))

    def interpolate(self, t: float, Vec u) -> None:
        """Interpolate the solution to a given time.

        Collective.

        Parameters
        ----------
        t
            The time to interpolate.
        u
            The state vector to interpolate.

        See Also
        --------
        petsc.TSInterpolate

        """
        cdef PetscReal rval = asReal(t)
        CHKERR(TSInterpolate(self.ts, rval, u.vec))

    def setStepLimits(self, hmin: float, hmax: float) -> None:
        """Set the minimum and maximum allowed step sizes.

        Logically collective.

        Parameters
        ----------
        hmin
            the minimum step size
        hmax
            the maximum step size

        See Also
        --------
        petsc.TSAdaptSetStepLimits

        """
        cdef PetscTSAdapt tsadapt = NULL
        cdef PetscReal hminr = toReal(hmin)
        cdef PetscReal hmaxr = toReal(hmax)
        TSGetAdapt(self.ts, &tsadapt)
        CHKERR(TSAdaptSetStepLimits(tsadapt, hminr, hmaxr))

    def getStepLimits(self) -> tuple[float, float]:
        """Return the minimum and maximum allowed time step sizes.

        Not collective.

        See Also
        --------
        petsc.TSAdaptGetStepLimits

        """
        cdef PetscTSAdapt tsadapt = NULL
        cdef PetscReal hminr = 0.
        cdef PetscReal hmaxr = 0.
        TSGetAdapt(self.ts, &tsadapt)
        CHKERR(TSAdaptGetStepLimits(tsadapt, &hminr, &hmaxr))
        return (asReal(hminr), asReal(hmaxr))

    # --- Adjoint methods ---

    def setSaveTrajectory(self) -> None:
        """Enable to save solutions as an internal `TS` trajectory.

        Collective.

        This routine should be called after all `TS` options have
        been set.

        Notes
        -----
        ``-ts_save_trajectory`` can be used to save a trajectory to a file.

        See Also
        --------
        petsc.TSSetSaveTrajectory

        """
        CHKERR(TSSetSaveTrajectory(self.ts))

    def removeTrajectory(self) -> None:
        """Remove the internal `TS` trajectory object.

        Collective.

        See Also
        --------
        petsc.TSRemoveTrajectory

        """
        CHKERR(TSRemoveTrajectory(self.ts))

    def getCostIntegral(self) -> Vec:
        """Return a vector of values of the integral term in the cost functions.

        Not collective.

        See Also
        --------
        petsc.TSGetCostIntegral

        """
        cdef Vec cost = Vec()
        CHKERR(TSGetCostIntegral(self.ts, &cost.vec))
        CHKERR(PetscINCREF(cost.obj))
        return cost

    def setCostGradients(
        self,
        vl: Vec | Sequence[Vec] | None,
        vm: Vec | Sequence[Vec] | None = None) -> None:
        """Set the cost gradients.

        Logically collective.

        Parameters
        ----------
        vl
            gradients with respect to the initial condition variables, the
            dimension and parallel layout of these vectors is the same as the
            ODE solution vector
        vm
            gradients with respect to the parameters, the number of entries in
            these vectors is the same as the number of parameters

        See Also
        --------
        petsc.TSSetCostGradients

        """
        cdef PetscInt n = 0
        cdef PetscVec *vecl = NULL
        cdef PetscVec *vecm = NULL
        cdef mem1 = None, mem2 = None
        if isinstance(vl, Vec): vl = [vl]
        if isinstance(vm, Vec): vm = [vm]
        if vl is not None:
            n = <PetscInt>len(vl)
        elif vm is not None:
            n = <PetscInt>len(vm)
        if vl is not None:
            assert len(vl) == <Py_ssize_t>n
            mem1 = oarray_p(empty_p(<PetscInt>n), NULL, <void**>&vecl)
            for i from 0 <= i < n:
                vecl[i] = (<Vec?>vl[i]).vec
        if vm is not None:
            assert len(vm) == <Py_ssize_t>n
            mem2 = oarray_p(empty_p(<PetscInt>n), NULL, <void**>&vecm)
            for i from 0 <= i < n:
                vecm[i] = (<Vec?>vm[i]).vec
        self.set_attr('__costgradients_memory', (mem1, mem2))
        CHKERR(TSSetCostGradients(self.ts, n, vecl, vecm))

    def getCostGradients(self) -> tuple[list[Vec], list[Vec]]:
        """Return the cost gradients.

        Not collective.

        See Also
        --------
        setCostGradients, petsc.TSGetCostGradients

        """
        cdef PetscInt n = 0
        cdef PetscVec *vecl = NULL
        cdef PetscVec *vecm = NULL
        CHKERR(TSGetCostGradients(self.ts, &n, &vecl, &vecm))
        cdef object vl = None, vm = None
        if vecl != NULL:
            vl = [ref_Vec(vecl[i]) for i from 0 <= i < n]
        if vecm != NULL:
            vm = [ref_Vec(vecm[i]) for i from 0 <= i < n]
        return (vl, vm)

    def setRHSJacobianP(
        self,
        jacobianp: TSRHSJacobianP | None,
        Mat A=None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the function that computes the Jacobian with respect to the parameters.

        Logically collective.

        Parameters
        ----------
        jacobianp
            The user-defined function.
        A
            The matrix into which the Jacobian will be computed.
        args
            Additional positional arguments for ``jacobianp``.
        kargs
            Additional keyword arguments for ``jacobianp``.

        See Also
        --------
        petsc.TSSetRHSJacobianP

        """
        cdef PetscMat Amat=NULL
        if A is not None: Amat = A.mat
        if jacobianp is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (jacobianp, args, kargs)
            self.set_attr('__rhsjacobianp__', context)
            CHKERR(TSSetRHSJacobianP(self.ts, Amat, TS_RHSJacobianP, <void*>context))
        else:
            CHKERR(TSSetRHSJacobianP(self.ts, Amat, NULL, NULL))

    def createQuadratureTS(self, forward: bool = True) -> TS:
        """Create a sub `TS` that evaluates integrals over time.

        Collective.

        Parameters
        ----------
        forward
            Enable to evaluate forward in time.

        See Also
        --------
        petsc.TSCreateQuadratureTS

        """
        cdef TS qts = TS()
        cdef PetscBool fwd = forward
        CHKERR(TSCreateQuadratureTS(self.ts, fwd, &qts.ts))
        CHKERR(PetscINCREF(qts.obj))
        return qts

    def getQuadratureTS(self) -> tuple[bool, TS]:
        """Return the sub `TS` that evaluates integrals over time.

        Not collective.

        Returns
        -------
        forward : bool
            True if evaluating the integral forward in time
        qts : TS
            The sub `TS`

        See Also
        --------
        petsc.TSGetQuadratureTS

        """
        cdef TS qts = TS()
        cdef PetscBool fwd = PETSC_FALSE
        CHKERR(TSGetQuadratureTS(self.ts, &fwd, &qts.ts))
        CHKERR(PetscINCREF(qts.obj))
        return (toBool(fwd), qts)

    def setRHSJacobianP(
        self,
        rhsjacobianp: TSRHSJacobianP | None,
        Mat A=None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the function that computes the Jacobian with respect to the parameters.

        Collective.

        Parameters
        ----------
        rhsjacobianp
            The function to compute the Jacobian
        A
            The JacobianP matrix
        args
            Additional positional arguments for ``rhsjacobianp``.
        kargs
            Additional keyword arguments for ``rhsjacobianp``.

        See Also
        --------
        petsc.TSSetRHSJacobianP

        """
        cdef PetscMat Amat=NULL
        if A is not None: Amat = A.mat
        if rhsjacobianp is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (rhsjacobianp, args, kargs)
            self.set_attr('__rhsjacobianp__', context)
            CHKERR(TSSetRHSJacobianP(self.ts, Amat, TS_RHSJacobianP, <void*>context))
        else:
            CHKERR(TSSetRHSJacobianP(self.ts, Amat, NULL, NULL))

    def computeRHSJacobianP(self, t: float, Vec x, Mat J) -> None:
        """Run the user-defined JacobianP function.

        Collective.

        Parameters
        ----------
        t
            The time at which to compute the Jacobian.
        x
            The solution at which to compute the Jacobian.
        J
            The output Jacobian matrix.

        See Also
        --------
        petsc.TSComputeRHSJacobianP

        """
        cdef PetscReal rval = asReal(t)
        CHKERR(TSComputeRHSJacobianP(self.ts, rval, x.vec, J.mat))

    def adjointSetSteps(self, adjoint_steps: int) -> None:
        """Set the number of steps the adjoint solver should take backward in time.

        Logically collective.

        Parameters
        ----------
        adjoint_steps
            The number of steps to take.

        See Also
        --------
        petsc.TSAdjointSetSteps

        """
        cdef PetscInt ival = asInt(adjoint_steps)
        CHKERR(TSAdjointSetSteps(self.ts, ival))

    def adjointSetUp(self) -> None:
        """Set up the internal data structures for the later use of an adjoint solver.

        Collective.

        See Also
        --------
        petsc.TSAdjointSetUp

        """
        CHKERR(TSAdjointSetUp(self.ts))

    def adjointSolve(self) -> None:
        """Solve the discrete adjoint problem for an ODE/DAE.

        Collective.

        See Also
        --------
        petsc.TSAdjointSolve

        """
        CHKERR(TSAdjointSolve(self.ts))

    def adjointStep(self) -> None:
        """Step one time step backward in the adjoint run.

        Collective.

        See Also
        --------
        petsc.TSAdjointStep

        """
        CHKERR(TSAdjointStep(self.ts))

    def adjointReset(self) -> None:
        """Reset a `TS`, removing any allocated vectors and matrices.

        Collective.

        See Also
        --------
        petsc.TSAdjointReset

        """
        CHKERR(TSAdjointReset(self.ts))

    # --- Python ---

    def createPython(self, context: Any = None, comm: Comm | None = None) -> Self:
        """Create an integrator of Python type.

        Collective.

        Parameters
        ----------
        context
            An instance of the Python class implementing the required methods.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc_python_ts, setType, setPythonContext, Type.PYTHON

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscTS newts = NULL
        CHKERR(TSCreate(ccomm, &newts))
        CHKERR(PetscCLEAR(self.obj)); self.ts = newts
        CHKERR(TSSetType(self.ts, TSPYTHON))
        CHKERR(TSPythonSetContext(self.ts, <void*>context))
        return self

    def setPythonContext(self, context: Any) -> None:
        """Set the instance of the class implementing the required Python methods.

        Not collective.

        See Also
        --------
        petsc_python_ts, getPythonContext

        """
        CHKERR(TSPythonSetContext(self.ts, <void*>context))

    def getPythonContext(self) -> Any:
        """Return the instance of the class implementing the required Python methods.

        Not collective.

        See Also
        --------
        petsc_python_ts, setPythonContext

        """
        cdef void *context = NULL
        CHKERR(TSPythonGetContext(self.ts, &context))
        if context == NULL: return None
        else: return <object> context

    def setPythonType(self, py_type: str) -> None:
        """Set the fully qualified Python name of the class to be used.

        Collective.

        See Also
        --------
        petsc_python_ts, setPythonContext, getPythonType, petsc.TSPythonSetType

        """
        cdef const char *cval = NULL
        py_type = str2bytes(py_type, &cval)
        CHKERR(TSPythonSetType(self.ts, cval))

    def getPythonType(self) -> str:
        """Return the fully qualified Python name of the class used by the solver.

        Not collective.

        See Also
        --------
        petsc_python_ts, setPythonContext, setPythonType, petsc.TSPythonGetType

        """
        cdef const char *cval = NULL
        CHKERR(TSPythonGetType(self.ts, &cval))
        return bytes2str(cval)

    # --- Theta ---

    def setTheta(self, theta: float) -> None:
        """Set the abscissa of the stage in ``(0, 1]`` for `Type.THETA`.

        Logically collective.

        Parameters
        ----------
        theta
            stage abscissa

        Notes
        -----
        ``-ts_theta_theta`` can be used to set a value from the commandline.

        See Also
        --------
        petsc.TSThetaSetTheta

        """
        cdef PetscReal rval = asReal(theta)
        CHKERR(TSThetaSetTheta(self.ts, rval))

    def getTheta(self) -> float:
        """Return the abscissa of the stage in ``(0, 1]`` for `Type.THETA`.

        Not collective.

        See Also
        --------
        petsc.TSThetaGetTheta

        """
        cdef PetscReal rval = 0
        CHKERR(TSThetaGetTheta(self.ts, &rval))
        return toReal(rval)

    def setThetaEndpoint(self, flag=True) -> None:
        """Set to use the endpoint variant of `Type.THETA`.

        Logically collective.

        Parameters
        ----------
        flag
            Enable to use the endpoint variant.

        See Also
        --------
        petsc.TSThetaSetEndpoint

        """
        cdef PetscBool bval = flag
        CHKERR(TSThetaSetEndpoint(self.ts, bval))

    def getThetaEndpoint(self) -> bool:
        """Return whether the endpoint variable of `Type.THETA` is used.

        Not collective.

        See Also
        --------
        petsc.TSThetaGetEndpoint

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(TSThetaGetEndpoint(self.ts, &flag))
        return toBool(flag)

    # --- Alpha ---

    def setAlphaRadius(self, radius: float) -> None:
        """Set the spectral radius for `Type.ALPHA`.

        Logically collective.

        Parameters
        ----------
        radius
            the spectral radius

        Notes
        -----
        ``-ts_alpha_radius`` can be used to set this from the commandline.

        See Also
        --------
        petsc.TSAlphaSetRadius

        """
        cdef PetscReal rval = asReal(radius)
        CHKERR(TSAlphaSetRadius(self.ts, rval))

    def setAlphaParams(
        self,
        alpha_m: float | None = None,
        alpha_f: float | None = None,
        gamma: float | None = None) -> None:
        """Set the algorithmic parameters for `Type.ALPHA`.

        Logically collective.

        Users should call `setAlphaRadius`.

        Parameters
        ----------
        alpha_m
            Parameter, leave `None`  to keep current value.
        alpha_f
            Parameter, leave `None`  to keep current value.
        gamma
            Parameter, leave `None`  to keep current value.

        See Also
        --------
        petsc.TSAlphaSetParams

        """
        cdef PetscReal rval1 = 0, rval2 = 0, rval3 = 0
        try: CHKERR(TSAlphaGetParams(self.ts, &rval1, &rval2, &rval3))
        except PetscError: pass
        if alpha_m is not None: rval1 = asReal(alpha_m)
        if alpha_f is not None: rval2 = asReal(alpha_f)
        if gamma   is not None: rval3 = asReal(gamma)
        CHKERR(TSAlphaSetParams(self.ts,  rval1,  rval2,  rval3))

    def getAlphaParams(self) -> tuple[float, float, float]:
        """Return the algorithmic parameters for `Type.ALPHA`.

        Not collective.

        See Also
        --------
        petsc.TSAlphaGetParams

        """
        cdef PetscReal rval1 = 0, rval2 = 0, rval3 = 0
        CHKERR(TSAlphaGetParams(self.ts, &rval1, &rval2, &rval3))
        return (toReal(rval1), toReal(rval2), toReal(rval3))

    # --- application context ---

    property appctx:
        """Application context."""
        def __get__(self) -> Any:
            return self.getAppCtx()

        def __set__(self, value) -> None:
            self.setAppCtx(value)

    # --- discretization space ---

    property dm:
        """The `DM`."""
        def __get__(self) -> DM:
            return self.getDM()

        def __set__(self, value) -> None:
            self.setDM(value)

    # --- xxx ---

    property problem_type:
        """The problem type."""
        def __get__(self) -> ProblemType:
            return self.getProblemType()

        def __set__(self, value) -> None:
            self.setProblemType(value)

    property equation_type:
        """The equation type."""
        def __get__(self) -> EquationType:
            return self.getEquationType()

        def __set__(self, value) -> None:
            self.setEquationType(value)

    property snes:
        """The `SNES`."""
        def __get__(self) -> SNES:
            return self.getSNES()

    property ksp:
        """The `KSP`."""
        def __get__(self) -> KSP:
            return self.getKSP()

    property vec_sol:
        """The solution vector."""
        def __get__(self) -> Vec:
            return self.getSolution()

    # --- xxx ---

    property time:
        """The current time."""
        def __get__(self) -> float:
            return self.getTime()

        def __set__(self, value) -> None:
            self.setTime(value)

    property time_step:
        """The current time step size."""
        def __get__(self) -> None:
            return self.getTimeStep()

        def __set__(self, value):
            self.setTimeStep(value)

    property step_number:
        """The current step number."""
        def __get__(self) -> int:
            return self.getStepNumber()

        def __set__(self, value) -> None:
            self.setStepNumber(value)

    property max_time:
        """The maximum time."""
        def __get__(self) -> float:
            return self.getMaxTime()

        def __set__(self, value) -> None:
            self.setMaxTime(value)

    property max_steps:
        """The maximum number of steps."""
        def __get__(self) -> int:
            return self.getMaxSteps()

        def __set__(self, value) -> None:
            self.setMaxSteps(value)

    # --- convergence ---

    property rtol:
        """The relative tolerance."""
        def __get__(self) -> float:
            return self.getTolerances()[0]

        def __set__(self, value) -> None:
            self.setTolerances(rtol=value)

    property atol:
        """The absolute tolerance."""
        def __get__(self) -> float:
            return self.getTolerances()[1]

        def __set__(self, value) -> None:
            self.setTolerances(atol=value)

    property reason:
        """The converged reason."""
        def __get__(self) -> ConvergedReason:
            return self.getConvergedReason()

        def __set__(self, value) -> None:
            self.setConvergedReason(value)

    property iterating:
        """Indicates the `TS` is still iterating."""
        def __get__(self) -> bool:
            return self.reason == 0

    property converged:
        """Indicates the `TS` has converged."""
        def __get__(self) -> bool:
            return self.reason > 0

    property diverged:
        """Indicates the `TS` has stopped."""
        def __get__(self) -> bool:
            return self.reason < 0

# -----------------------------------------------------------------------------

del TSType
del TSRKType
del TSARKIMEXType
del TSDIRKType
del TSProblemType
del TSEquationType
del TSExactFinalTime
del TSConvergedReason

# -----------------------------------------------------------------------------
