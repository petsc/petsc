cdef class DMShell(DM):
    """A shell DM object, used to manage user-defined problem data."""

    def create(self, comm: Comm | None = None) -> Self:
        """Creates a shell DM object.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.DMShellCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR(DMShellCreate(ccomm, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm
        return self

    def setMatrix(self, Mat mat) -> None:
        """Set a template matrix.

        Collective.

        Parameters
        ----------
        mat
            The template matrix.

        See Also
        --------
        petsc.DMShellSetMatrix

        """
        CHKERR(DMShellSetMatrix(self.dm, mat.mat))

    def setGlobalVector(self, Vec gv) -> None:
        """Set a template global vector.

        Logically collective.

        Parameters
        ----------
        gv
            Template vector.

        See Also
        --------
        setLocalVector, petsc.DMShellSetGlobalVector

        """
        CHKERR(DMShellSetGlobalVector(self.dm, gv.vec))

    def setLocalVector(self, Vec lv) -> None:
        """Set a template local vector.

        Logically collective.

        Parameters
        ----------
        lv
            Template vector.

        See Also
        --------
        setGlobalVector, petsc.DMShellSetLocalVector

        """
        CHKERR(DMShellSetLocalVector(self.dm, lv.vec))

    def setCreateGlobalVector(
        self,
        create_gvec: Callable[[DM], Vec] | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the routine to create a global vector.

        Logically collective.

        Parameters
        ----------
        create_gvec
            The creation routine.
        args
            Additional positional arguments for ``create_gvec``.
        kargs
            Additional keyword arguments for ``create_gvec``.

        See Also
        --------
        setCreateLocalVector, petsc.DMShellSetCreateGlobalVector

        """
        if create_gvec is not None:
            if args  is None: args = ()
            if kargs is None: kargs = {}
            context = (create_gvec, args, kargs)
            self.set_attr('__create_global_vector__', context)
            CHKERR(DMShellSetCreateGlobalVector(self.dm, DMSHELL_CreateGlobalVector))
        else:
            CHKERR(DMShellSetCreateGlobalVector(self.dm, NULL))

    def setCreateLocalVector(
        self,
        create_lvec: Callable[[DM], Vec] | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the routine to create a local vector.

        Logically collective.

        Parameters
        ----------
        create_lvec
            The creation routine.
        args
            Additional positional arguments for ``create_lvec``.
        kargs
            Additional keyword arguments for ``create_lvec``.

        See Also
        --------
        setCreateGlobalVector, petsc.DMShellSetCreateLocalVector

        """
        if create_lvec is not None:
            if args  is None: args = ()
            if kargs is None: kargs = {}
            context = (create_lvec, args, kargs)
            self.set_attr('__create_local_vector__', context)
            CHKERR(DMShellSetCreateLocalVector(self.dm, DMSHELL_CreateLocalVector))
        else:
            CHKERR(DMShellSetCreateLocalVector(self.dm, NULL))

    def setGlobalToLocal(
        self,
        begin: Callable[[DM, Vec, InsertMode, Vec], None] | None,
        end: Callable[[DM, Vec, InsertMode, Vec], None] | None,
        begin_args: tuple[Any, ...] | None = None,
        begin_kargs: dict[str, Any] | None = None,
        end_args: tuple[Any, ...] | None = None,
        end_kargs: dict[str, Any] | None = None) -> None:
        """Set the routines used to perform a global to local scatter.

        Logically collective.

        Parameters
        ----------
        dm
            The `DMShell`.
        begin
            The routine which begins the global to local scatter.
        end
            The routine which ends the global to local scatter.
        begin_args
            Additional positional arguments for ``begin``.
        begin_kargs
            Additional keyword arguments for ``begin``.
        end_args
            Additional positional arguments for ``end``.
        end_kargs
            Additional keyword arguments for ``end``.

        See Also
        --------
        petsc.DMShellSetGlobalToLocal

        """
        cdef PetscDMShellXToYFunction cbegin = NULL, cend = NULL
        if begin is not None:
            if begin_args  is None: begin_args = ()
            if begin_kargs is None: begin_kargs = {}
            context = (begin, begin_args, begin_kargs)
            self.set_attr('__g2l_begin__', context)
            cbegin = &DMSHELL_GlobalToLocalBegin
        if end is not None:
            if end_args  is None: end_args = ()
            if end_kargs is None: end_kargs = {}
            context = (end, end_args, end_kargs)
            self.set_attr('__g2l_end__', context)
            cend = &DMSHELL_GlobalToLocalEnd
        CHKERR(DMShellSetGlobalToLocal(self.dm, cbegin, cend))

    def setGlobalToLocalVecScatter(self, Scatter gtol) -> None:
        """Set a `Scatter` context for global to local communication.

        Logically collective.

        Parameters
        ----------
        gtol
            The global to local `Scatter` context.

        See Also
        --------
        petsc.DMShellSetGlobalToLocalVecScatter

        """
        CHKERR(DMShellSetGlobalToLocalVecScatter(self.dm, gtol.sct))

    def setLocalToGlobal(
        self,
        begin: Callable[[DM, Vec, InsertMode, Vec], None] | None,
        end: Callable[[DM, Vec, InsertMode, Vec], None] | None,
        begin_args: tuple[Any, ...] | None = None,
        begin_kargs: dict[str, Any] | None = None,
        end_args: tuple[Any, ...] | None = None,
        end_kargs: dict[str, Any] | None = None) -> None:
        """Set the routines used to perform a local to global scatter.

        Logically collective.

        Parameters
        ----------
        begin
            The routine which begins the local to global scatter.
        end
            The routine which ends the local to global scatter.
        begin_args
            Additional positional arguments for ``begin``.
        begin_kargs
            Additional keyword arguments for ``begin``.
        end_args
            Additional positional arguments for ``end``.
        end_kargs
            Additional keyword arguments for ``end``.

        See Also
        --------
        petsc.DMShellSetLocalToGlobal

        """
        cdef PetscDMShellXToYFunction cbegin = NULL, cend = NULL
        if begin is not None:
            if begin_args  is None: begin_args = ()
            if begin_kargs is None: begin_kargs = {}
            context = (begin, begin_args, begin_kargs)
            self.set_attr('__l2g_begin__', context)
            cbegin = &DMSHELL_LocalToGlobalBegin
        if end is not None:
            if end_args  is None: end_args = ()
            if end_kargs is None: end_kargs = {}
            context = (end, end_args, end_kargs)
            self.set_attr('__l2g_end__', context)
            cend = &DMSHELL_LocalToGlobalEnd
        CHKERR(DMShellSetLocalToGlobal(self.dm, cbegin, cend))

    def setLocalToGlobalVecScatter(self, Scatter ltog) -> None:
        """Set a `Scatter` context for local to global communication.

        Logically collective.

        Parameters
        ----------
        ltog
            The local to global `Scatter` context.

        See Also
        --------
        petsc.DMShellSetLocalToGlobalVecScatter

        """
        CHKERR(DMShellSetLocalToGlobalVecScatter(self.dm, ltog.sct))

    def setLocalToLocal(
        self,
        begin: Callable[[DM, Vec, InsertMode, Vec], None] | None,
        end: Callable[[DM, Vec, InsertMode, Vec], None] | None,
        begin_args: tuple[Any, ...] | None = None,
        begin_kargs: dict[str, Any] | None = None,
        end_args: tuple[Any, ...] | None = None,
        end_kargs: dict[str, Any] | None = None) -> None:
        """Set the routines used to perform a local to local scatter.

        Logically collective.

        Parameters
        ----------
        begin
            The routine which begins the local to local scatter.
        end
            The routine which ends the local to local scatter.
        begin_args
            Additional positional arguments for ``begin``.
        begin_kargs
            Additional keyword arguments for ``begin``.
        end_args
            Additional positional arguments for ``end``.
        end_kargs
            Additional keyword arguments for ``end``.

        See Also
        --------
        petsc.DMShellSetLocalToLocal

        """
        cdef PetscDMShellXToYFunction cbegin = NULL, cend = NULL
        cbegin = NULL
        cend = NULL
        if begin is not None:
            if begin_args  is None: begin_args = ()
            if begin_kargs is None: begin_kargs = {}
            context = (begin, begin_args, begin_kargs)
            self.set_attr('__l2l_begin__', context)
            cbegin = &DMSHELL_LocalToLocalBegin
        if end is not None:
            if end_args  is None: end_args = ()
            if end_kargs is None: end_kargs = {}
            context = (end, end_args, end_kargs)
            self.set_attr('__l2l_end__', context)
            cend = &DMSHELL_LocalToLocalEnd
        CHKERR(DMShellSetLocalToLocal(self.dm, cbegin, cend))

    def setLocalToLocalVecScatter(self, Scatter ltol) -> None:
        """Set a ``Scatter`` context for local to local communication.

        Logically collective.

        Parameters
        ----------
        ltol
            The local to local ``Scatter`` context.

        See Also
        --------
        petsc.DMShellSetLocalToLocalVecScatter

        """
        CHKERR(DMShellSetLocalToLocalVecScatter(self.dm, ltol.sct))

    def setCreateMatrix(
        self,
        create_matrix: Callable[[DM], Mat] | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the routine to create a matrix.

        Logically collective.

        Parameters
        ----------
        create_matrix
            The function to create a matrix.
        args
            Additional positional arguments for ``create_matrix``.
        kargs
            Additional keyword arguments for ``create_matrix``.

        See Also
        --------
        petsc.DMShellSetCreateMatrix

        """
        if create_matrix is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (create_matrix, args, kargs)
            self.set_attr('__create_matrix__', context)
            CHKERR(DMShellSetCreateMatrix(self.dm, DMSHELL_CreateMatrix))
        else:
            CHKERR(DMShellSetCreateMatrix(self.dm, NULL))

    def setCoarsen(
        self,
        coarsen: Callable[[DM, Comm], DM] | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the routine used to coarsen the `DMShell`.

        Logically collective.

        Parameters
        ----------
        coarsen
            The routine which coarsens the DM.
        args
            Additional positional arguments for ``coarsen``.
        kargs
            Additional keyword arguments for ``coarsen``.

        See Also
        --------
        setRefine, petsc.DMShellSetCoarsen

        """
        if coarsen is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (coarsen, args, kargs)
            self.set_attr('__coarsen__', context)
            CHKERR(DMShellSetCoarsen(self.dm, DMSHELL_Coarsen))
        else:
            CHKERR(DMShellSetCoarsen(self.dm, NULL))

    def setRefine(
        self,
        refine: Callable[[DM, Comm], DM] | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the routine used to refine the `DMShell`.

        Logically collective.

        Parameters
        ----------
        refine
            The routine which refines the DM.
        args
            Additional positional arguments for ``refine``.
        kargs
            Additional keyword arguments for ``refine``.

        See Also
        --------
        setCoarsen, petsc.DMShellSetRefine

        """
        if refine is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (refine, args, kargs)
            self.set_attr('__refine__', context)
            CHKERR(DMShellSetRefine(self.dm, DMSHELL_Refine))
        else:
            CHKERR(DMShellSetRefine(self.dm, NULL))

    def setCreateInterpolation(
        self,
        create_interpolation: Callable[[DM, DM], tuple[Mat, Vec]] | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the routine used to create the interpolation operator.

        Logically collective.

        Parameters
        ----------
        create_interpolation
            The routine to create the interpolation.
        args
            Additional positional arguments for ``create_interpolation``.
        kargs
            Additional keyword arguments for ``create_interpolation``.

        See Also
        --------
        petsc.DMShellSetCreateInterpolation

        """
        if create_interpolation is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (create_interpolation, args, kargs)
            self.set_attr('__create_interpolation__', context)
            CHKERR(DMShellSetCreateInterpolation(self.dm, DMSHELL_CreateInterpolation))
        else:
            CHKERR(DMShellSetCreateInterpolation(self.dm, NULL))

    def setCreateInjection(
        self,
        create_injection: Callable[[DM, DM], Mat] | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the routine used to create the injection operator.

        Logically collective.

        Parameters
        ----------
        create_injection
            The routine to create the injection.
        args
            Additional positional arguments for ``create_injection``.
        kargs
            Additional keyword arguments for ``create_injection``.

        See Also
        --------
        petsc.DMShellSetCreateInjection

        """
        if create_injection is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (create_injection, args, kargs)
            self.set_attr('__create_injection__', context)
            CHKERR(DMShellSetCreateInjection(self.dm, DMSHELL_CreateInjection))
        else:
            CHKERR(DMShellSetCreateInjection(self.dm, NULL))

    def setCreateRestriction(
        self,
        create_restriction: Callable[[DM, DM], Mat] | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the routine used to create the restriction operator.

        Logically collective.

        Parameters
        ----------
        create_restriction
            The routine to create the restriction
        args
            Additional positional arguments for ``create_restriction``.
        kargs
            Additional keyword arguments for ``create_restriction``.

        See Also
        --------
        petsc.DMShellSetCreateRestriction

        """
        if create_restriction is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (create_restriction, args, kargs)
            self.set_attr('__create_restriction__', context)
            CHKERR(DMShellSetCreateRestriction(self.dm, DMSHELL_CreateRestriction))
        else:
            CHKERR(DMShellSetCreateRestriction(self.dm, NULL))

    def setCreateFieldDecomposition(
        self,
        decomp: Callable[[DM], tuple[list[str] | None, list[IS] | None, list[DM] | None]] | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the routine used to create a field decomposition.

        Logically collective.

        Parameters
        ----------
        decomp
            The routine to create the decomposition.
        args
            Additional positional arguments for ``decomp``.
        kargs
            Additional keyword arguments for ``decomp``.

        See Also
        --------
        petsc.DMShellSetCreateFieldDecomposition

        """
        if decomp is not None:
            if args  is None: args = ()
            if kargs is None: kargs = {}
            context = (decomp, args, kargs)
            self.set_attr('__create_field_decomp__', context)
            CHKERR(DMShellSetCreateFieldDecomposition(self.dm, DMSHELL_CreateFieldDecomposition))
        else:
            CHKERR(DMShellSetCreateFieldDecomposition(self.dm, NULL))

    def setCreateDomainDecomposition(
        self,
        decomp: Callable[[DM], tuple[list[str] | None, list[IS] | None, list[IS] | None, list[DM] | None]] | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the routine used to create a domain decomposition.

        Logically collective.

        Parameters
        ----------
        decomp
            The routine to create the decomposition.
        args
            Additional positional arguments for ``decomp``.
        kargs
            Additional keyword arguments for ``decomp``.

        See Also
        --------
        petsc.DMShellSetCreateDomainDecomposition

        """
        if decomp is not None:
            if args  is None: args = ()
            if kargs is None: kargs = {}
            context = (decomp, args, kargs)
            self.set_attr('__create_domain_decomp__', context)
            CHKERR(DMShellSetCreateDomainDecomposition(self.dm, DMSHELL_CreateDomainDecomposition))
        else:
            CHKERR(DMShellSetCreateDomainDecomposition(self.dm, NULL))

    def setCreateDomainDecompositionScatters(
        self,
        scatter: Callable[[DM, list[DM]], tuple[list[Scatter], list[Scatter], list[Scatter]]] | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the routine used to create the scatter contexts for domain decomposition.

        Logically collective.

        Parameters
        ----------
        scatter
            The routine to create the scatters.
        args
            Additional positional arguments for ``scatter``.
        kargs
            Additional keyword arguments for ``scatter``.

        See Also
        --------
        petsc.DMShellSetCreateDomainDecompositionScatters

        """
        if scatter is not None:
            if args  is None: args = ()
            if kargs is None: kargs = {}
            context = (scatter, args, kargs)
            self.set_attr('__create_domain_decomp_scatters__', context)
            CHKERR(DMShellSetCreateDomainDecompositionScatters(self.dm, DMSHELL_CreateDomainDecompositionScatters))
        else:
            CHKERR(DMShellSetCreateDomainDecompositionScatters(self.dm, NULL))

    def setCreateSubDM(
        self,
        create_subdm: Callable[[DM, Sequence[int]], tuple[IS, DM]] | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the routine used to create a sub DM from the `DMShell`.

        Logically collective.

        Parameters
        ----------
        subdm
            The routine to create the decomposition.
        args
            Additional positional arguments for ``subdm``.
        kargs
            Additional keyword arguments for ``subdm``.

        See Also
        --------
        petsc.DMShellSetCreateSubDM

        """
        if create_subdm is not None:
            if args  is None: args = ()
            if kargs is None: kargs = {}
            context = (create_subdm, args, kargs)
            self.set_attr('__create_subdm__', context)
            CHKERR(DMShellSetCreateSubDM(self.dm, DMSHELL_CreateSubDM))
        else:
            CHKERR(DMShellSetCreateSubDM(self.dm, NULL))
