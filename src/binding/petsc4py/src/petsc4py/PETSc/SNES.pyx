# --------------------------------------------------------------------

class SNESType(object):
    """SNES solver type.

    See Also
    --------
    petsc.SNESType

    """
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
    """SNES norm schedule.

    See Also
    --------
    petsc.SNESNormSchedule

    """
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

# FIXME Missing reference petsc.SNESConvergedReason
class SNESConvergedReason(object):
    """SNES solver termination reason.

    See Also
    --------
    petsc.SNESGetConvergedReason

    """
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
    """Nonlinear equations solver.

    SNES is described in the `PETSc manual <petsc:manual/snes>`.

    See Also
    --------
    petsc.SNES

    """

    Type = SNESType
    NormSchedule = SNESNormSchedule
    ConvergedReason = SNESConvergedReason

    # --- xxx ---

    def __cinit__(self):
        self.obj  = <PetscObject*> &self.snes
        self.snes = NULL

    # --- xxx ---

    def view(self, Viewer viewer=None) -> None:
        """View the solver.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        petsc.SNESView

        """
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( SNESView(self.snes, cviewer) )

    def destroy(self) -> Self:
        """Destroy the solver.

        Collective.

        See Also
        --------
        petsc.SNESDestroy

        """
        CHKERR( SNESDestroy(&self.snes) )
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create a SNES solver.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        Sys.getDefaultComm, petsc.SNESCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSNES newsnes = NULL
        CHKERR( SNESCreate(ccomm, &newsnes) )
        PetscCLEAR(self.obj); self.snes = newsnes
        return self

    def setType(self, snes_type: Type | str) -> None:
        """Set the type of the solver.

        Logically collective.

        Parameters
        ----------
        snes_type
            The type of the solver.

        See Also
        --------
        getType, petsc.SNESSetType

        """
        cdef PetscSNESType cval = NULL
        snes_type = str2bytes(snes_type, &cval)
        CHKERR( SNESSetType(self.snes, cval) )

    def getType(self) -> str:
        """Return the type of the solver.

        Not collective.

        See Also
        --------
        setType, petsc.SNESGetType

        """
        cdef PetscSNESType cval = NULL
        CHKERR( SNESGetType(self.snes, &cval) )
        return bytes2str(cval)

    def setOptionsPrefix(self, prefix: str) -> None:
        """Set the prefix used for searching for options in the database.

        Logically collective.

        See Also
        --------
        petsc_options, petsc.SNESSetOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( SNESSetOptionsPrefix(self.snes, cval) )

    def getOptionsPrefix(self) -> str:
        """Return the prefix used for searching for options in the database.

        Not collective.

        See Also
        --------
        petsc_options, setOptionsPrefix, petsc.SNESGetOptionsPrefix

        """
        cdef const char *cval = NULL
        CHKERR( SNESGetOptionsPrefix(self.snes, &cval) )
        return bytes2str(cval)

    def appendOptionsPrefix(self, prefix: str) -> None:
        """Append to the prefix used for searching for options in the database.

        Logically collective.

        See Also
        --------
        petsc_options, setOptionsPrefix, petsc.SNESAppendOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( SNESAppendOptionsPrefix(self.snes, cval) )

    def setFromOptions(self) -> None:
        """Configure the solver from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.SNESSetFromOptions

        """
        CHKERR( SNESSetFromOptions(self.snes) )

    # --- application context ---

    def setApplicationContext(self, appctx: Any) -> None:
        """Set the application context."""
        self.set_attr('__appctx__', appctx)
        if appctx is not None:
            registerAppCtx(<void*>appctx)
            CHKERR( SNESSetApplicationContext(self.snes, <void*>appctx) )
        else:
            CHKERR( SNESSetApplicationContext(self.snes, NULL) )

    def getApplicationContext(self) -> Any:
        """Return the application context."""
        cdef void *ctx
        appctx = self.get_attr('__appctx__')
        if appctx is None:
            CHKERR( SNESGetApplicationContext(self.snes, &ctx) )
            appctx = toAppCtx(ctx)
        return appctx

    # backward compatibility
    setAppCtx = setApplicationContext
    getAppCtx = getApplicationContext

    # --- discretization space ---

    def getDM(self) -> DM:
        """Return the `DM` associated with the solver.

        Not collective.

        See Also
        --------
        setDM, petsc.SNESGetDM

        """
        cdef PetscDM newdm = NULL
        CHKERR( SNESGetDM(self.snes, &newdm) )
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        PetscINCREF(dm.obj)
        return dm

    def setDM(self, DM dm) -> None:
        """Associate a `DM` with the solver.

        Not collective.

        See Also
        --------
        getDM, petsc.SNESSetDM

        """
        CHKERR( SNESSetDM(self.snes, dm.dm) )

    # --- FAS ---

    def setFASInterpolation(self, level: int, Mat mat) -> None:
        """Set the `Mat` to be used to apply the interpolation from level-1 to level.

        Collective.

        See Also
        --------
        getFASInterpolation, setFASRestriction, setFASInjection
        petsc.SNESFASSetInterpolation, petsc.SNESFAS

        """
        cdef PetscInt clevel = asInt(level)
        CHKERR( SNESFASSetInterpolation(self.snes, clevel, mat.mat) )

    def getFASInterpolation(self, level: int) -> Mat:
        """Return the `Mat` used to apply the interpolation from level-1 to level.

        Not collective.

        See Also
        --------
        setFASInterpolation, petsc.SNESFASGetInterpolation, petsc.SNESFAS

        """
        cdef PetscInt clevel = asInt(level)
        cdef Mat mat = Mat()
        CHKERR( SNESFASGetInterpolation(self.snes, clevel, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

    def setFASRestriction(self, level: int, Mat mat) -> None:
        """Set the `Mat` to be used to apply the restriction from level-1 to level.

        Collective.

        See Also
        --------
        setFASRScale, getFASRestriction, setFASInterpolation, setFASInjection
        petsc.SNESFASSetRestriction, petsc.SNESFAS

        """
        cdef PetscInt clevel = asInt(level)
        CHKERR( SNESFASSetRestriction(self.snes, clevel, mat.mat) )

    def getFASRestriction(self, level: int) -> Mat:
        """Return the `Mat` used to apply the restriction from level-1 to level.

        Not collective.

        See Also
        --------
        setFASRestriction, petsc.SNESFASGetRestriction, petsc.SNESFAS

        """
        cdef PetscInt clevel = asInt(level)
        cdef Mat mat = Mat()
        CHKERR( SNESFASGetRestriction(self.snes, clevel, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

    def setFASInjection(self, level: int, Mat mat) -> None:
        """Set the `Mat` to be used to apply the injection from level-1 to level.

        Collective.

        See Also
        --------
        getFASInjection, setFASInterpolation, setFASRestriction
        petsc.SNESFASSetInjection, petsc.SNESFAS

        """
        cdef PetscInt clevel = asInt(level)
        CHKERR( SNESFASSetInjection(self.snes, clevel, mat.mat) )

    def getFASInjection(self, level: int) -> Mat:
        """Return the `Mat` used to apply the injection from level-1 to level.

        Not collective.

        See Also
        --------
        setFASInjection, petsc.SNESFASGetInjection, petsc.SNESFAS

        """
        cdef PetscInt clevel = asInt(level)
        cdef Mat mat = Mat()
        CHKERR( SNESFASGetInjection(self.snes, clevel, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

    def setFASRScale(self, level: int, Vec vec) -> None:
        """Set the scaling factor of the restriction operator from level to level-1.

        Collective.

        See Also
        --------
        setFASRestriction, petsc.SNESFASSetRScale, petsc.SNESFAS

        """
        cdef PetscInt clevel = asInt(level)
        CHKERR( SNESFASSetRScale(self.snes, clevel, vec.vec) )

    def setFASLevels(self, levels: int, comms: Sequence[Comm] = None) -> None:
        """Set the number of levels to use with FAS.

        Collective.

        Parameters
        ----------
        levels
            The number of levels
        comms
            An optional sequence of communicators of length `levels`, or `None` for the default communicator `Sys.getDefaultComm`.

        See Also
        --------
        getFASLevels, petsc.SNESFASSetLevels, petsc.SNESFAS

        """
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

    def getFASLevels(self) -> int:
        """Return the number of levels used.

        Not collective.

        See Also
        --------
        setFASLevels, petsc.SNESFASGetLevels, petsc.SNESFAS

        """
        cdef PetscInt levels = 0
        CHKERR( SNESFASGetLevels(self.snes, &levels) )
        return toInt(levels)

    def getFASCycleSNES(self, level: int) -> SNES:
        """Return the `SNES` corresponding to a particular level of the FAS hierarchy.

        Not collective.

        See Also
        --------
        setFASLevels, getFASCoarseSolve, getFASSmoother
        petsc.SNESFASGetCycleSNES, petsc.SNESFAS

        """
        cdef PetscInt clevel = asInt(level)
        cdef SNES lsnes = SNES()
        CHKERR( SNESFASGetCycleSNES(self.snes, clevel, &lsnes.snes) )
        PetscINCREF(lsnes.obj)
        return lsnes

    def getFASCoarseSolve(self) -> SNES:
        """Return the `SNES` used at the coarsest level of the FAS hierarchy.

        Not collective.

        See Also
        --------
        getFASSmoother, petsc.SNESFASGetCoarseSolve, petsc.SNESFAS

        """
        cdef SNES smooth = SNES()
        CHKERR( SNESFASGetCoarseSolve(self.snes, &smooth.snes) )
        PetscINCREF(smooth.obj)
        return smooth

    def getFASSmoother(self, level: int) -> SNES:
        """Return the smoother used at a given level of the FAS hierarchy.

        Not collective.

        See Also
        --------
        setFASLevels, getFASCoarseSolve, getFASSmootherDown, getFASSmootherUp
        petsc.SNESFASGetSmoother, petsc.SNESFAS

        """
        cdef PetscInt clevel = asInt(level)
        cdef SNES smooth = SNES()
        CHKERR( SNESFASGetSmoother(self.snes, clevel, &smooth.snes) )
        PetscINCREF(smooth.obj)
        return smooth

    def getFASSmootherDown(self, level: int) -> SNES:
        """Return the downsmoother used at a given level of the FAS hierarchy.

        Not collective.

        See Also
        --------
        setFASLevels, getFASCoarseSolve, getFASSmoother, getFASSmootherUp
        petsc.SNESFASGetSmootherDown, petsc.SNESFAS

        """
        cdef PetscInt clevel = asInt(level)
        cdef SNES smooth = SNES()
        CHKERR( SNESFASGetSmootherDown(self.snes, clevel, &smooth.snes) )
        PetscINCREF(smooth.obj)
        return smooth

    def getFASSmootherUp(self, level: int) -> SNES:
        """Return the upsmoother used at a given level of the FAS hierarchy.

        Not collective.

        See Also
        --------
        setFASLevels, getFASCoarseSolve, getFASSmoother, getFASSmootherDown
        petsc.SNESFASGetSmootherUp, petsc.SNESFAS

        """
        cdef PetscInt clevel = asInt(level)
        cdef SNES smooth = SNES()
        CHKERR( SNESFASGetSmootherUp(self.snes, clevel, &smooth.snes) )
        PetscINCREF(smooth.obj)
        return smooth

    # --- nonlinear preconditioner ---

    def getNPC(self) -> SNES:
        """Return the nonlinear preconditioner associated with the solver.

        Not collective.

        See Also
        --------
        setNPC, hasNPC, setNPCSide, getNPCSide, petsc.SNESGetNPC

        """
        cdef SNES snes = SNES()
        CHKERR( SNESGetNPC(self.snes, &snes.snes) )
        PetscINCREF(snes.obj)
        return snes

    def hasNPC(self) -> bool:
        """Return a boolean indicating whether the solver has a nonlinear preconditioner.

        Not collective.

        See Also
        --------
        setNPC, getNPC, setNPCSide, getNPCSide, petsc.SNESHasNPC

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( SNESHasNPC(self.snes, &flag) )
        return toBool(flag)

    def setNPC(self, SNES snes) -> None:
        """Set the nonlinear preconditioner.

        Logically collective.

        See Also
        --------
        getNPC, hasNPC, setNPCSide, getNPCSide, petsc.SNESSetNPC

        """
        CHKERR( SNESSetNPC(self.snes, snes.snes) )

    def setNPCSide(self, side: PC.Side) -> None:
        """Set the nonlinear preconditioning side.

        Collective.

        See Also
        --------
        setNPC, getNPC, hasNPC, getNPCSide, petsc.SNESSetNPCSide

        """
        CHKERR( SNESSetNPCSide(self.snes, side) )

    def getNPCSide(self) -> PC.Side:
        """Return the nonlinear preconditioning side.

        Not collective.

        See Also
        --------
        setNPC, getNPC, hasNPC, setNPCSide, petsc.SNESGetNPCSide

        """
        cdef PetscPCSide side = PC_RIGHT
        CHKERR( SNESGetNPCSide(self.snes, &side) )
        return side

    # --- user Function/Jacobian routines ---

    def setLineSearchPreCheck(self, precheck: SNESLSPreFunction,
                              args: tuple[Any, ...] | None = None,
                              kargs: dict[str, Any] | None = None) -> None:
        """Set the callback that will be called before applying the linesearch.

        Logically collective.

        Parameters
        ----------
        precheck
            The callback.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        See Also
        --------
        petsc.SNESLineSearchSetPreCheck

        """
        cdef PetscSNESLineSearch snesls = NULL
        CHKERR( SNESGetLineSearch(self.snes, &snesls) )
        if precheck is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (precheck, args, kargs)
            self.set_attr('__precheck__', context)
            # FIXME callback
            CHKERR( SNESLineSearchSetPreCheck(snesls, SNES_PreCheck, <void*> context) )
        else:
            self.set_attr('__precheck__', None)
            CHKERR( SNESLineSearchSetPreCheck(snesls, NULL, NULL) )

    def setInitialGuess(self, initialguess: SNESGuessFunction,
                        args: tuple[Any, ...] | None = None,
                        kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute the initial guess.

        Logically collective.

        Parameters
        ----------
        initialguess
            The callback.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        See Also
        --------
        getInitialGuess, petsc.SNESSetComputeInitialGuess

        """
        if initialguess is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (initialguess, args, kargs)
            self.set_attr('__initialguess__', context)
            CHKERR( SNESSetComputeInitialGuess(self.snes, SNES_InitialGuess, <void*>context) )
        else:
            self.set_attr('__initialguess__', None)
            CHKERR( SNESSetComputeInitialGuess(self.snes, NULL, NULL) )

    def getInitialGuess(self) -> SNESGuessFunction:
        """Return the callback to compute the initial guess.

        See Also
        --------
        setInitialGuess

        """
        return self.get_attr('__initialguess__')

    def setFunction(self, function: SNESFunction, Vec f=None,
                    args: tuple[Any, ...] | None = None,
                    kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute the nonlinear function.

        Logically collective.

        Parameters
        ----------
        function
            The callback.
        f
            An optional vector to store the result.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        See Also
        --------
        getFunction, petsc.SNESSetFunction

        """
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

    def getFunction(self) -> SNESFunction:
        """Return the callback to compute the nonlinear function.

        Not collective.

        See Also
        --------
        setFunction, petsc.SNESGetFunction

        """
        cdef Vec f = Vec()
        cdef void* ctx
        cdef PetscErrorCode (*fun)(PetscSNES,PetscVec,PetscVec,void*)
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

    def setUpdate(self, update: SNESUpdateFunction,
                  args: tuple[Any, ...] | None = None,
                  kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute update at the beginning of the nonlinear step.

        Logically collective.

        Parameters
        ----------
        update
            The callback.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        See Also
        --------
        getUpdate, petsc.SNESSetUpdate

        """
        if update is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (update, args, kargs)
            self.set_attr('__update__', context)
            CHKERR( SNESSetUpdate(self.snes, SNES_Update) )
        else:
            self.set_attr('__update__', None)
            CHKERR( SNESSetUpdate(self.snes, NULL) )

    def getUpdate(self) -> SNESUpdateFunction:
        """Return the callback to compute the update at the beginning of the nonlinear step.

        Not collective.

        See Also
        --------
        setUpdate

        """
        return self.get_attr('__update__')

    def setJacobian(self, jacobian: SNESJacobianFunction, Mat J=None, Mat P=None,
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
        getJacobian, petsc.SNESSetJacobian

        """
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

    def getJacobian(self) -> tuple[Mat, Mat, SNESJacobianFunction]:
        """Return the matrices used to compute the Jacobian and the callback tuple.

        Not collective.

        Returns
        -------
        J : Mat
            The matrix to store the Jacobian.
        P : Mat
            The matrix to construct the preconditioner.
        callback : SNESJacobianFunction
            callback, positional and keyword arguments.

        See Also
        --------
        setJacobian, petsc.SNESGetJacobian

        """
        cdef Mat J = Mat()
        cdef Mat P = Mat()
        CHKERR( SNESGetJacobian(self.snes, &J.mat, &P.mat, NULL, NULL) )
        PetscINCREF(J.obj)
        PetscINCREF(P.obj)
        cdef object jacobian = self.get_attr('__jacobian__')
        return (J, P, jacobian)

    def setObjective(self, objective: SNESObjFunction,
                     args: tuple[Any, ...] | None = None,
                     kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute the objective function.

        Logically collective.

        Parameters
        ----------
        objective
            The Jacobian callback.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        See Also
        --------
        getObjective, petsc.SNESSetObjective

        """
        if objective is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (objective, args, kargs)
            self.set_attr('__objective__', context)
            CHKERR( SNESSetObjective(self.snes, SNES_Objective, <void*>context) )
        else:
            CHKERR( SNESSetObjective(self.snes, NULL, NULL) )

    def getObjective(self) -> SNESObjFunction:
        """Return the objective callback tuple.

        Not collective.

        See Also
        --------
        setObjective

        """
        CHKERR( SNESGetObjective(self.snes, NULL, NULL) )
        cdef object objective = self.get_attr('__objective__')
        return objective

    def computeFunction(self, Vec x, Vec f) -> None:
        """Compute the function.

        Collective.

        Parameters
        ----------
        x
            The input state vector.
        f
            The output vector.

        See Also
        --------
        setFunction, petsc.SNESComputeFunction

        """
        CHKERR( SNESComputeFunction(self.snes, x.vec, f.vec) )

    def computeJacobian(self, Vec x, Mat J, Mat P=None) -> None:
        """Compute the Jacobian.

        Collective.

        Parameters
        ----------
        x
            The input state vector.
        J
            The output Jacobian matrix.
        P
            The output Jacobian matrix used to construct the preconditioner.

        See Also
        --------
        setJacobian, petsc.SNESComputeJacobian

        """
        cdef PetscMat jmat = J.mat, pmat = J.mat
        if P is not None: pmat = P.mat
        CHKERR( SNESComputeJacobian(self.snes, x.vec, jmat, pmat) )

    def computeObjective(self, Vec x) -> float:
        """Compute the value of the objective function.

        Collective.

        Parameters
        ----------
        x
            The input state vector.

        See Also
        --------
        setObjective, petsc.SNESComputeObjective

        """
        cdef PetscReal o = 0
        CHKERR( SNESComputeObjective(self.snes, x.vec, &o) )
        return toReal(o)

    def setNGS(self, ngs: SNESNGSFunction,
               args: tuple[Any, ...] | None = None,
               kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute nonlinear Gauss-Seidel.

        Logically collective.

        Parameters
        ----------
        ngs
            The nonlinear Gauss-Seidel callback.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        See Also
        --------
        getNGS, computeNGS, petsc.SNESSetNGS

        """
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (ngs, args, kargs)
        self.set_attr('__ngs__', context)
        CHKERR( SNESSetNGS(self.snes, SNES_NGS, <void*>context) )

    def getNGS(self) -> SNESNGSFunction:
        """Return the nonlinear Gauss-Seidel callback tuple.

        Not collective.

        See Also
        --------
        setNGS, computeNGS

        """
        CHKERR( SNESGetNGS(self.snes, NULL, NULL) )
        cdef object ngs = self.get_attr('__ngs__')
        return ngs

    def computeNGS(self, Vec x, Vec b=None) -> None:
        """Compute a nonlinear Gauss-Seidel step.

        Collective.

        Parameters
        ----------
        x
            The input/output state vector.
        b
            The input right-hand side vector.

        See Also
        --------
        setNGS, getNGS, petsc.SNESComputeNGS

        """
        cdef PetscVec bvec = NULL
        if b is not None: bvec = b.vec
        CHKERR( SNESComputeNGS(self.snes, bvec, x.vec) )

    # --- tolerances and convergence ---

    def setTolerances(self, rtol: float = None, atol: float = None, stol: float = None, max_it: int = None) -> None:
        """Set the tolerance parameters used in the solver convergence tests.

        Collective.

        Parameters
        ----------
        rtol
            The relative norm of the residual. Defaults to `DEFAULT`.
        atol
            The absolute norm of the residual. Defaults to `DEFAULT`.
        stol
            The absolute norm of the step. Defaults to `DEFAULT`.
        max_it
            The maximum allowed number of iterations. Defaults to `DEFAULT`

        See Also
        --------
        getTolerances, petsc.SNESSetTolerances

        """
        cdef PetscReal crtol, catol, cstol
        crtol = catol = cstol = PETSC_DEFAULT
        cdef PetscInt cmaxit = PETSC_DEFAULT
        if rtol   is not None: crtol  = asReal(rtol)
        if atol   is not None: catol  = asReal(atol)
        if stol   is not None: cstol  = asReal(stol)
        if max_it is not None: cmaxit = asInt(max_it)
        CHKERR( SNESSetTolerances(self.snes, catol, crtol, cstol,
                                  cmaxit, PETSC_DEFAULT) )

    def getTolerances(self) -> tuple[float, float, float, int]:
        """Return the tolerance parameters used in the solver convergence tests.

        Collective.

        Returns
        -------
        rtol : float
            The relative norm of the residual.
        atol : float
            The absolute norm of the residual.
        stol : float
            The absolute norm of the step.
        max_it : int
            The maximum allowed number of iterations.

        See Also
        --------
        setTolerances, petsc.SNESGetTolerances

        """
        cdef PetscReal crtol=0, catol=0, cstol=0
        cdef PetscInt cmaxit=0
        CHKERR( SNESGetTolerances(self.snes, &catol, &crtol, &cstol,
                                  &cmaxit, NULL) )
        return (toReal(crtol), toReal(catol), toReal(cstol), toInt(cmaxit))

    def setNormSchedule(self, normsched: NormSchedule) -> None:
        """Set the norm schedule.

        Collective.

        See Also
        --------
        getNormSchedule, petsc.SNESSetNormSchedule

        """
        CHKERR( SNESSetNormSchedule(self.snes, normsched) )

    def getNormSchedule(self) -> NormSchedule:
        """Return the norm schedule.

        Not collective.

        See Also
        --------
        setNormSchedule, petsc.SNESGetNormSchedule

        """
        cdef PetscSNESNormSchedule normsched = SNES_NORM_NONE
        CHKERR( SNESGetNormSchedule(self.snes, &normsched) )
        return normsched

    def setConvergenceTest(self, converged: SNESConvergedFunction | Literal["skip", "default"],
                           args: tuple[Any, ...] | None = None,
                           kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to use as convergence test.

        Logically collective.

        Parameters
        ----------
        converged
            The convergence testing callback.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        See Also
        --------
        getConvergenceTest, callConvergenceTest, petsc.SNESSetConvergenceTest

        """
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

    def getConvergenceTest(self) -> SNESConvergedFunction:
        """Return the callback to used as convergence test.

        Not collective.

        See Also
        --------
        setConvergenceTest, callConvergenceTest

        """
        return self.get_attr('__converged__')

    def callConvergenceTest(self, its: int, xnorm: float, ynorm: float, fnorm: float) -> ConvergedReason:
        """Compute the convergence test.

        Collective.

        Parameters
        ----------
        its
            Iteration number.
        xnorm
            Solution norm.
        ynorm
            Update norm.
        fnorm
            Function norm.

        See Also
        --------
        setConvergenceTest, getConvergenceTest

        """
        cdef PetscInt  ival  = asInt(its)
        cdef PetscReal rval1 = asReal(xnorm)
        cdef PetscReal rval2 = asReal(ynorm)
        cdef PetscReal rval3 = asReal(fnorm)
        cdef PetscSNESConvergedReason reason = SNES_CONVERGED_ITERATING
        CHKERR( SNESConvergenceTestCall(self.snes, ival,
                                        rval1, rval2, rval3, &reason) )
        return reason

    def setConvergenceHistory(self, length=None, reset=False) -> None:
        """Set the convergence history.

        Logically collective.

        See Also
        --------
        petsc.SNESSetConvergenceHistory

        """
        cdef PetscReal *rdata = NULL
        cdef PetscInt  *idata = NULL
        cdef PetscInt   size = 1000
        cdef PetscBool flag = PETSC_FALSE
        #FIXME
        if   length is True:     pass
        elif length is not None: size = asInt(length)
        if size < 0: size = 1000
        if reset: flag = PETSC_TRUE
        cdef object rhist = oarray_r(empty_r(size), NULL, &rdata)
        cdef object ihist = oarray_i(empty_i(size), NULL, &idata)
        self.set_attr('__history__', (rhist, ihist))
        CHKERR( SNESSetConvergenceHistory(self.snes, rdata, idata, size, flag) )

    def getConvergenceHistory(self) -> tuple[ArrayReal, ArrayInt]:
        """Return the convergence history.

        Not collective.

        See Also
        --------
        petsc.SNESGetConvergenceHistory

        """
        cdef PetscReal *rdata = NULL
        cdef PetscInt  *idata = NULL
        cdef PetscInt   size = 0
        CHKERR( SNESGetConvergenceHistory(self.snes, &rdata, &idata, &size) )
        cdef object rhist = array_r(size, rdata)
        cdef object ihist = array_i(size, idata)
        return (rhist, ihist)

    def logConvergenceHistory(self, norm: float, linear_its: int = 0) -> None:
        """Log residual norm and linear iterations."""
        cdef PetscReal rval = asReal(norm)
        cdef PetscInt  ival = asInt(linear_its)
        CHKERR( SNESLogConvergenceHistory(self.snes, rval, ival) )

    def setResetCounters(self, reset: bool = True) -> None:
        """Set the flag to reset the counters.

        Collective.

        See Also
        --------
        petsc.SNESSetCountersReset

        """
        cdef PetscBool flag = reset
        CHKERR( SNESSetCountersReset(self.snes, flag) )

    # --- monitoring ---

    def setMonitor(self, monitor: SNESMonitorFunction, args: tuple[Any, ...] | None = None, kargs: dict[str, Any] | None = None) -> None:
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
        getMonitor, petsc.SNESMonitorSet

        """
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

    def getMonitor(self) -> list[tuple[SNESMonitorFunction, tuple[Any, ...], dict[str, Any]]]:
        """Return the callback used to monitor solver convergence.

        Not collective.

        See Also
        --------
        setMonitor

        """
        return self.get_attr('__monitor__')

    def monitorCancel(self) -> None:
        """Cancel all the monitors of the solver.

        Logically collective.

        See Also
        --------
        setMonitor, petsc.SNESMonitorCancel

        """
        CHKERR( SNESMonitorCancel(self.snes) )
        self.set_attr('__monitor__', None)

    cancelMonitor = monitorCancel

    def monitor(self, its, rnorm) -> None:
        """Monitor the solver.

        Collective.

        Parameters
        ----------
        its
            Current number of iterations.
        rnorm
            Current value of the residual norm.

        See Also
        --------
        setMonitor, petsc.SNESMonitor

        """
        cdef PetscInt  ival = asInt(its)
        cdef PetscReal rval = asReal(rnorm)
        CHKERR( SNESMonitor(self.snes, ival, rval) )

    # --- more tolerances ---

    def setMaxFunctionEvaluations(self, max_funcs: int) -> None:
        """Set the maximum allowed number of function evaluations.

        Collective.

        See Also
        --------
        getMaxFunctionEvaluations, petsc.SNESSetTolerances

        """
        cdef PetscReal r = PETSC_DEFAULT
        cdef PetscInt  i = PETSC_DEFAULT
        cdef PetscInt ival = asInt(max_funcs)
        CHKERR( SNESSetTolerances(self.snes, r, r, r, i, ival) )

    def getMaxFunctionEvaluations(self) -> int:
        """Return the maximum allowed number of function evaluations.

        Not collective.

        See Also
        --------
        setMaxFunctionEvaluations, petsc.SNESSetTolerances

        """
        cdef PetscReal *r = NULL
        cdef PetscInt  *i = NULL
        cdef PetscInt ival = 0
        CHKERR( SNESGetTolerances(self.snes, r, r, r, i, &ival) )
        return toInt(ival)

    def getFunctionEvaluations(self) -> int:
        """Return the current number of function evaluations.

        Not collective.

        See Also
        --------
        setMaxFunctionEvaluations, petsc.SNESGetNumberFunctionEvals

        """
        cdef PetscInt ival = 0
        CHKERR( SNESGetNumberFunctionEvals(self.snes, &ival) )
        return toInt(ival)

    def setMaxStepFailures(self, max_fails: int) -> None:
        """Set the maximum allowed number of step failures.

        Collective.

        See Also
        --------
        getMaxStepFailures, petsc.SNESSetMaxNonlinearStepFailures

        """
        cdef PetscInt ival = asInt(max_fails)
        CHKERR( SNESSetMaxNonlinearStepFailures(self.snes, ival) )

    def getMaxStepFailures(self) -> int:
        """Return the maximum allowed number of step failures.

        Not collective.

        See Also
        --------
        setMaxStepFailures, petsc.SNESGetMaxNonlinearStepFailures

        """
        cdef PetscInt ival = 0
        CHKERR( SNESGetMaxNonlinearStepFailures(self.snes, &ival) )
        return toInt(ival)

    def getStepFailures(self) -> int:
        """Return the current number of step failures.

        Not collective.

        See Also
        --------
        getMaxStepFailures, petsc.SNESGetNonlinearStepFailures

        """
        cdef PetscInt ival = 0
        CHKERR( SNESGetNonlinearStepFailures(self.snes, &ival) )
        return toInt(ival)

    def setMaxKSPFailures(self, max_fails: int) -> None:
        """Set the maximum allowed number of linear solve failures.

        Collective.

        See Also
        --------
        getMaxKSPFailures, petsc.SNESSetMaxLinearSolveFailures

        """
        cdef PetscInt ival = asInt(max_fails)
        CHKERR( SNESSetMaxLinearSolveFailures(self.snes, ival) )

    def getMaxKSPFailures(self) -> int:
        """Return the maximum allowed number of linear solve failures.

        Not collective.

        See Also
        --------
        setMaxKSPFailures, petsc.SNESGetMaxLinearSolveFailures

        """
        cdef PetscInt ival = 0
        CHKERR( SNESGetMaxLinearSolveFailures(self.snes, &ival) )
        return toInt(ival)

    def getKSPFailures(self) -> int:
        """Return the current number of linear solve failures.

        Not collective.

        See Also
        --------
        getMaxKSPFailures, petsc.SNESGetLinearSolveFailures

        """
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

    def setUp(self) -> None:
        """Set up the internal data structures for using the solver.

        Collective.

        See Also
        --------
        petsc.SNESSetUp

        """
        CHKERR( SNESSetUp(self.snes) )

    def reset(self) -> None:
        """Reset the solver.

        Collective.

        See Also
        --------
        petsc.SNESReset

        """
        CHKERR( SNESReset(self.snes) )

    def solve(self, Vec b = None, Vec x = None) -> None:
        """Solve the nonlinear equations.

        Collective.

        Parameters
        ----------
        b
            The affine right-hand side or `None` to use zero.
        x
            The starting vector or `None` to use the vector stored internally.

        See Also
        --------
        setSolution, getSolution, petsc.SNESSolve

        """
        cdef PetscVec rhs = NULL
        cdef PetscVec sol = NULL
        if b is not None: rhs = b.vec
        if x is not None: sol = x.vec
        CHKERR( SNESSolve(self.snes, rhs, sol) )

    def setConvergedReason(self, reason: ConvergedReason) -> None:
        """Set the termination flag.

        Collective.

        See Also
        --------
        getConvergedReason, petsc.SNESSetConvergedReason

        """
        cdef PetscSNESConvergedReason eval = reason
        CHKERR( SNESSetConvergedReason(self.snes, eval) )

    def getConvergedReason(self) -> ConvergedReason:
        """Return the termination flag.

        Not collective.

        See Also
        --------
        setConvergedReason, petsc.SNESGetConvergedReason

        """
        cdef PetscSNESConvergedReason reason = SNES_CONVERGED_ITERATING
        CHKERR( SNESGetConvergedReason(self.snes, &reason) )
        return reason

    def setErrorIfNotConverged(self, flag: bool) -> None:
        """Immediately generate an error if the solver has not converged.

        Collective.

        See Also
        --------
        getErrorIfNotConverged, petsc.SNESSetErrorIfNotConverged

        """
        cdef PetscBool ernc = asBool(flag)
        CHKERR( SNESSetErrorIfNotConverged(self.snes, ernc) )

    def getErrorIfNotConverged(self) -> bool:
        """Return the flag indicating error on divergence.

        Not collective.

        See Also
        --------
        setErrorIfNotConverged, petsc.SNESGetErrorIfNotConverged

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( SNESGetErrorIfNotConverged(self.snes, &flag) )
        return toBool(flag)

    def setIterationNumber(self, its: int) -> None:
        """Set the current iteration number. This is only of use to implementers of custom SNES types.

        Collective.

        See Also
        --------
        getIterationNumber, petsc.SNESSetIterationNumber

        """
        cdef PetscInt ival = asInt(its)
        CHKERR( SNESSetIterationNumber(self.snes, ival) )

    def getIterationNumber(self) -> int:
        """Return the current iteration number.

        Not collective.

        See Also
        --------
        setIterationNumber, petsc.SNESGetIterationNumber

        """
        cdef PetscInt ival = 0
        CHKERR( SNESGetIterationNumber(self.snes, &ival) )
        return toInt(ival)

    def setForceIteration(self, force: bool) -> None:
        """Force solve to take at least one iteration.

        Collective.

        See Also
        --------
        petsc.SNESSetForceIteration

        """
        cdef PetscBool bval = asBool(force)
        CHKERR( SNESSetForceIteration(self.snes, bval) )

    def setFunctionNorm(self, norm: float) -> None:
        """Set the function norm value. This is only of use to implementers of custom SNES types.

        Collective.

        See Also
        --------
        getFunctionNorm, petsc.SNESSetFunctionNorm

        """
        cdef PetscReal rval = asReal(norm)
        CHKERR( SNESSetFunctionNorm(self.snes, rval) )

    def getFunctionNorm(self) -> float:
        """Return the function norm.

        Not collective.

        See Also
        --------
        setFunctionNorm, petsc.SNESGetFunctionNorm

        """
        cdef PetscReal rval = 0
        CHKERR( SNESGetFunctionNorm(self.snes, &rval) )
        return toReal(rval)

    def getLinearSolveIterations(self) -> int:
        """Return the total number of linear iterations.

        Not collective.

        See Also
        --------
        petsc.SNESGetLinearSolveIterations

        """
        cdef PetscInt ival = 0
        CHKERR( SNESGetLinearSolveIterations(self.snes, &ival) )
        return toInt(ival)

    def getRhs(self) -> Vec:
        """Return the vector holding the right-hand side.

        Not collective.

        See Also
        --------
        petsc.SNESGetRhs

        """
        cdef Vec vec = Vec()
        CHKERR( SNESGetRhs(self.snes, &vec.vec) )
        PetscINCREF(vec.obj)
        return vec

    def getSolution(self) -> Vec:
        """Return the vector holding the solution.

        Not collective.

        See Also
        --------
        setSolution, petsc.SNESGetSolution

        """
        cdef Vec vec = Vec()
        CHKERR( SNESGetSolution(self.snes, &vec.vec) )
        PetscINCREF(vec.obj)
        return vec

    def setSolution(self, Vec vec) -> None:
        """Set the vector used to store the solution.

        Collective.

        See Also
        --------
        getSolution, petsc.SNESSetSolution

        """
        CHKERR( SNESSetSolution(self.snes, vec.vec) )

    def getSolutionUpdate(self) -> Vec:
        """Return the vector holding the solution update.

        Not collective.

        See Also
        --------
        petsc.SNESGetSolutionUpdate

        """
        cdef Vec vec = Vec()
        CHKERR( SNESGetSolutionUpdate(self.snes, &vec.vec) )
        PetscINCREF(vec.obj)
        return vec

    # --- linear solver ---

    def setKSP(self, KSP ksp) -> None:
        """Set the linear solver that will be used by the nonlinear solver.

        Logically collective.

        See Also
        --------
        getKSP, petsc.SNESSetKSP

        """
        CHKERR( SNESSetKSP(self.snes, ksp.ksp) )

    def getKSP(self) -> KSP:
        """Return the linear solver used by the nonlinear solver.

        Not collective.

        See Also
        --------
        setKSP, petsc.SNESGetKSP

        """
        cdef KSP ksp = KSP()
        CHKERR( SNESGetKSP(self.snes, &ksp.ksp) )
        PetscINCREF(ksp.obj)
        return ksp

    def setUseEW(self, flag: bool = True, *targs: Any, **kargs: Any) -> None:
        """Tell the solver to use the Eisenstat-Walker trick.

        Logically collective.

        Parameters
        ----------
        flag
            Whether or not to use the Eisenstat-Walker trick.
        *targs
            Positional arguments for `setParamsEW`.
        **kargs
            Keyword arguments for `setParamsEW`.

        See Also
        --------
        getUseEW, setParamsEW, petsc.SNESKSPSetUseEW

        """
        cdef PetscBool bval = flag
        CHKERR( SNESKSPSetUseEW(self.snes, bval) )
        if targs or kargs: self.setParamsEW(*targs, **kargs)

    def getUseEW(self) -> bool:
        """Return the boolean flag indicating if the solver uses the Eisenstat-Walker trick.

        See Also
        --------
        setUseEW, setParamsEW, petsc.SNESKSPGetUseEW

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( SNESKSPGetUseEW(self.snes, &flag) )
        return toBool(flag)

    def setParamsEW(self,
                    version: int = None,
                    rtol_0: float = None,
                    rtol_max: float = None,
                    gamma: float = None,
                    alpha: float = None,
                    alpha2: float = None,
                    threshold: float = None) -> None:
        """Set the parameters for the Eisenstat and Walker trick.

        Logically collective.

        Parameters
        ----------
        version
            The version of the algorithm. Defaults to `DEFAULT`.
        rtol_0
            The initial relative residual norm. Defaults to `DEFAULT`.
        rtol_max
            The maximum relative residual norm. Defaults to `DEFAULT`.
        gamma
            Parameter. Defaults to `DEFAULT`.
        alpha
            Parameter. Defaults to `DEFAULT`.
        alpha2
            Parameter. Defaults to `DEFAULT`.
        threshold
            Parameter. Defaults to `DEFAULT`.

        See Also
        --------
        setUseEW, getParamsEW, petsc.SNESKSPSetParametersEW

        """
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

    def getParamsEW(self) -> dict[str, int | float]:
        """Get the parameters of the Eisenstat and Walker trick.

        Not collective.

        See Also
        --------
        setUseEW, setParamsEW, petsc.SNESKSPGetParametersEW

        """
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

    # --- matrix free / finite differences ---

    def setUseMF(self, flag=True) -> None:
        """Set the boolean flag indicating to use matrix-free finite-differencing.

        Logically collective.

        See Also
        --------
        getUseMF

        """
        cdef PetscBool bval = flag
        CHKERR( SNESSetUseMFFD(self.snes, bval) )

    def getUseMF(self) -> bool:
        """Return the boolean flag indicating whether the solver uses matrix-free finite-differencing.

        Not collective.

        See Also
        --------
        setUseMF

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( SNESGetUseMFFD(self.snes, &flag) )
        return toBool(flag)

    def setUseFD(self, flag=True) -> None:
        """Set the boolean flag indicating to use coloring finite-differencing for Jacobian assembly.

        Logically collective.

        See Also
        --------
        getUseFD

        """
        cdef PetscBool bval = flag
        CHKERR( SNESSetUseFDColoring(self.snes, bval) )

    def getUseFD(self) -> False:
        """Return the boolean flag indicating whether the solver uses color finite-differencing assembly of the Jacobian.

        Not collective.

        See Also
        --------
        setUseFD

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( SNESGetUseFDColoring(self.snes, &flag) )
        return toBool(flag)

    # --- VI ---

    def setVariableBounds(self, Vec xl, Vec xu) -> None:
        """Set the vector for the variable bounds.

        Collective.

        See Also
        --------
        petsc.SNESVISetVariableBounds

        """
        CHKERR( SNESVISetVariableBounds(self.snes, xl.vec, xu.vec) )

    def getVIInactiveSet(self) -> IS:
        """Return the index set for the inactive set.

        Not collective.

        See Also
        --------
        petsc.SNESVIGetInactiveSet

        """
        cdef IS inact = IS()
        CHKERR( SNESVIGetInactiveSet(self.snes, &inact.iset) )
        PetscINCREF(inact.obj)
        return inact

    # --- Python ---

    def createPython(self, context: Any = None, comm: Comm | None = None) -> Self:
        """Create a nonlinear solver of Python type.

        Collective.

        Parameters
        ----------
        context
            An instance of the Python class implementing the required methods.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc_python_snes, setType, setPythonContext, Type.PYTHON

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSNES newsnes = NULL
        CHKERR( SNESCreate(ccomm, &newsnes) )
        PetscCLEAR(self.obj); self.snes = newsnes
        CHKERR( SNESSetType(self.snes, SNESPYTHON) )
        CHKERR( SNESPythonSetContext(self.snes, <void*>context) )
        return self

    def setPythonContext(self, context: Any) -> None:
        """Set the instance of the Python class implementing the required Python methods.

        Not collective.

        See Also
        --------
        petsc_python_snes, getPythonContext

        """
        CHKERR( SNESPythonSetContext(self.snes, <void*>context) )

    def getPythonContext(self) -> Any:
        """Return the instance of the Python class implementing the required Python methods.

        Not collective.

        See Also
        --------
        petsc_python_snes, setPythonContext

        """
        cdef void *context = NULL
        CHKERR( SNESPythonGetContext(self.snes, &context) )
        if context == NULL: return None
        else: return <object> context

    def setPythonType(self, py_type: str) -> None:
        """Set the fully qualified Python name of the class to be used.

        Collective.

        See Also
        --------
        petsc_python_snes, setPythonContext, getPythonType
        petsc.SNESPythonSetType

        """
        cdef const char *cval = NULL
        py_type = str2bytes(py_type, &cval)
        CHKERR( SNESPythonSetType(self.snes, cval) )

    def getPythonType(self) -> str:
        """Return the fully qualified Python name of the class used by the solver.

        Not collective.

        See Also
        --------
        petsc_python_snes, setPythonContext, setPythonType
        petsc.SNESPythonGetType

        """
        cdef const char *cval = NULL
        CHKERR( SNESPythonGetType(self.snes, &cval) )
        return bytes2str(cval)

    # --- Composite ---

    def getCompositeSNES(self, int n) -> SNES:
        """Return the n-th solver in the composite.

        Not collective.

        See Also
        --------
        getCompositeNumber, petsc.SNESCompositeGetSNES, petsc.SNESCOMPOSITE

        """
        cdef PetscInt cn
        cdef SNES snes = SNES()
        cn = asInt(n)
        CHKERR( SNESCompositeGetSNES(self.snes, cn, &snes.snes) )
        PetscINCREF(snes.obj)
        return snes

    def getCompositeNumber(self) -> int:
        """Return the number of solvers in the composite.

        Not collective.

        See Also
        --------
        getCompositeSNES, petsc.SNESCompositeGetNumber, petsc.SNESCOMPOSITE

        """
        cdef PetscInt cn = 0
        CHKERR( SNESCompositeGetNumber(self.snes, &cn) )
        return toInt(cn)

    # --- NASM ---

    def getNASMSNES(self, n: int) -> SNES:
        """Return the n-th solver in NASM.

        Not collective.

        See Also
        --------
        getNASMNumber, petsc.SNESNASMGetSNES, petsc.SNESNASM

        """
        cdef PetscInt cn = asInt(n)
        cdef SNES snes = SNES()
        CHKERR( SNESNASMGetSNES(self.snes, cn, &snes.snes) )
        PetscINCREF(snes.obj)
        return snes

    def getNASMNumber(self) -> int:
        """Return the number of solvers in NASM.

        Not collective.

        See Also
        --------
        getNASMSNES, petsc.SNESNASMGetNumber, petsc.SNESNASM

        """
        cdef PetscInt cn = 0
        CHKERR( SNESNASMGetNumber(self.snes, &cn) )
        return toInt(cn)

    # --- Patch ---

    def setPatchCellNumbering(self, Section sec) -> None:
        """Set cell patch numbering."""
        CHKERR( SNESPatchSetCellNumbering(self.snes, sec.sec) )

    def setPatchDiscretisationInfo(self, dms, bs,
                                   cellNodeMaps,
                                   subspaceOffsets,
                                   ghostBcNodes,
                                   globalBcNodes) -> None:
        """Set patch discretisation information."""
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

    def setPatchComputeOperator(self, operator, args=None, kargs=None) -> None:
        """Set patch compute operator."""
        if args is  None: args  = ()
        if kargs is None: kargs = {}
        context = (operator, args, kargs)
        self.set_attr("__patch_compute_operator__", context)
        CHKERR( SNESPatchSetComputeOperator(self.snes, PCPatch_ComputeOperator, <void*>context) )

    def setPatchComputeFunction(self, function, args=None, kargs=None) -> None:
        """Set patch compute function."""
        if args is  None: args  = ()
        if kargs is None: kargs = {}
        context = (function, args, kargs)
        self.set_attr("__patch_compute_function__", context)
        CHKERR( SNESPatchSetComputeFunction(self.snes, PCPatch_ComputeFunction, <void*>context) )

    def setPatchConstructType(self, typ, operator=None, args=None, kargs=None) -> None:
        """Set patch construct type."""
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
        """Application context."""
        def __get__(self) -> Any:
            return self.getAppCtx()
        def __set__(self, value):
            self.setAppCtx(value)

    # --- discretization space ---

    property dm:
        """`DM`."""
        def __get__(self) -> DM:
            return self.getDM()
        def __set__(self, value):
            self.setDM(value)

    # --- nonlinear preconditioner ---

    property npc:
        """Nonlinear preconditioner."""
        def __get__(self) -> SNES:
            return self.getNPC()
        def __set__(self, value):
            self.setNPC(value)

    # --- vectors ---

    property vec_sol:
        """Solution vector."""
        def __get__(self) -> Vec:
            return self.getSolution()

    property vec_upd:
        """Update vector."""
        def __get__(self) -> Vec:
            return self.getSolutionUpdate()

    property vec_rhs:
        """Right-hand side vector."""
        def __get__(self) -> Vec:
            return self.getRhs()

    # --- linear solver ---

    property ksp:
        """Linear solver."""
        def __get__(self) -> KSP:
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
        """Relative residual tolerance."""
        def __get__(self) -> float:
            return self.getTolerances()[0]
        def __set__(self, value):
            self.setTolerances(rtol=value)

    property atol:
        """Absolute residual tolerance."""
        def __get__(self) -> float:
            return self.getTolerances()[1]
        def __set__(self, value):
            self.setTolerances(atol=value)

    property stol:
        """Solution update tolerance."""
        def __get__(self) -> float:
            return self.getTolerances()[2]
        def __set__(self, value):
            self.setTolerances(stol=value)

    property max_it:
        """Maximum number of iterations."""
        def __get__(self) -> int:
            return self.getTolerances()[3]
        def __set__(self, value):
            self.setTolerances(max_it=value)

    # --- more tolerances ---

    property max_funcs:
        """Maximum number of function evaluations."""
        def __get__(self) -> int:
            return self.getMaxFunctionEvaluations()
        def __set__(self, value):
            self.setMaxFunctionEvaluations(value)

    # --- iteration ---

    property its:
        """Number of iterations."""
        def __get__(self) -> int:
            return self.getIterationNumber()
        def __set__(self, value):
            self.setIterationNumber(value)

    property norm:
        """Function norm."""
        def __get__(self) -> float:
            return self.getFunctionNorm()
        def __set__(self, value):
            self.setFunctionNorm(value)

    property history:
        """Convergence history."""
        def __get__(self) -> tuple[ArrayReal, ArrayInt]:
            return self.getConvergenceHistory()

    # --- convergence ---

    property reason:
        """Converged reason."""
        def __get__(self) -> ConvergedReason:
            return self.getConvergedReason()
        def __set__(self, value):
            self.setConvergedReason(value)

    property iterating:
        """Boolean indicating if the solver has not converged yet."""
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

    # --- matrix free / finite differences ---

    property use_mf:
        """Boolean indicating if the solver uses matrix-free finite-differencing."""
        def __get__(self) -> bool:
            return self.getUseMF()
        def __set__(self, value):
            self.setUseMF(value)

    property use_fd:
        """Boolean indicating if the solver uses coloring finite-differencing."""
        def __get__(self) -> bool:
            return self.getUseFD()
        def __set__(self, value):
            self.setUseFD(value)

# --------------------------------------------------------------------

del SNESType
del SNESNormSchedule
del SNESConvergedReason

# --------------------------------------------------------------------
