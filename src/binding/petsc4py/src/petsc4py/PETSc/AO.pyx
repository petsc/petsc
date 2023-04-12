# --------------------------------------------------------------------

class AOType(object):
    BASIC          = S_(AOBASIC)
    ADVANCED       = S_(AOADVANCED)
    MAPPING        = S_(AOMAPPING)
    MEMORYSCALABLE = S_(AOMEMORYSCALABLE)

# --------------------------------------------------------------------

cdef class AO(Object):
    """Application ordering object."""
    Type = AOType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.ao
        self.ao = NULL

    def view(self, Viewer viewer=None) -> None:
        """Display the application ordering.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` to display the ordering.

        See Also
        --------
        petsc.AOView

        """
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( AOView(self.ao, cviewer) )

    def destroy(self) -> Self:
        """Destroy the application ordering.

        Collective.

        See Also
        --------
        petsc.AODestroy

        """
        CHKERR( AODestroy(&self.ao) )
        return self

    def createBasic(
        self,
        app: Sequence[int] | IS,
        petsc: Sequence[int] | IS | None = None,
        comm: Comm | None = None,
    ) -> Self:
        """Return a basic application ordering using two orderings.

        Collective.

        The arrays/indices ``app`` and ``petsc`` must contain all the integers
        ``0`` to ``len(app)-1`` with no duplicates; that is there cannot be any
        "holes" in the indices. Use ``createMapping`` if you wish to have
        "holes" in the indices.

        Parameters
        ----------
        app
            The application ordering.
        petsc
            Another ordering (may be `None` to indicate the natural ordering,
            that is 0, 1, 2, 3, ...).
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        createMemoryScalable, createMapping, petsc.AOCreateBasicIS
        petsc.AOCreateBasic

        """
        cdef PetscIS isapp = NULL, ispetsc = NULL
        cdef PetscInt napp = 0, *idxapp = NULL,
        cdef PetscInt npetsc = 0, *idxpetsc = NULL
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscAO newao = NULL
        if isinstance(app, IS):
            isapp = (<IS>app).iset
            if petsc is not None:
                ispetsc = (<IS?>petsc).iset
            CHKERR( AOCreateBasicIS(isapp, ispetsc, &newao) )
        else:
            app = iarray_i(app, &napp, &idxapp)
            if petsc is not None:
                petsc = iarray_i(petsc, &npetsc, &idxpetsc)
                assert napp == npetsc, "incompatible array sizes"
            CHKERR( AOCreateBasic(ccomm, napp, idxapp, idxpetsc, &newao) )
        PetscCLEAR(self.obj); self.ao = newao
        return self

    def createMemoryScalable(
        self,
        app: Sequence[int] | IS,
        petsc: Sequence[int] | IS | None = None,
        comm: Comm | None = None,
    ) -> Self:
        """Return a memory scalable application ordering using two orderings.

        Collective.

        The arrays/indices ``app`` and ``petsc`` must contain all the integers
        ``0`` to ``len(app)-1`` with no duplicates; that is there cannot be any
        "holes" in the indices. Use ``createMapping`` if you wish to have
        "holes" in the indices.

        Comparing with ``createBasic``, this routine trades memory with message
        communication.

        Parameters
        ----------
        app
            The application ordering.
        petsc
            Another ordering (may be `None` to indicate the natural ordering,
            that is 0, 1, 2, 3, ...).
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        createBasic, createMapping, petsc.AOCreateMemoryScalableIS
        petsc.AOCreateMemoryScalable

        """
        cdef PetscIS isapp = NULL, ispetsc = NULL
        cdef PetscInt napp = 0, *idxapp = NULL,
        cdef PetscInt npetsc = 0, *idxpetsc = NULL
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscAO newao = NULL
        if isinstance(app, IS):
            isapp = (<IS>app).iset
            if petsc is not None:
                ispetsc = (<IS?>petsc).iset
            CHKERR( AOCreateMemoryScalableIS(isapp, ispetsc, &newao) )
        else:
            app = iarray_i(app, &napp, &idxapp)
            if petsc is not None:
                petsc = iarray_i(petsc, &npetsc, &idxpetsc)
                assert napp == npetsc, "incompatible array sizes"
            CHKERR( AOCreateMemoryScalable(ccomm, napp, idxapp, idxpetsc, &newao) )
        PetscCLEAR(self.obj); self.ao = newao
        return self

    def createMapping(
        self,
        app: Sequence[int] | IS,
        petsc: Sequence[int] | IS | None = None,
        comm: Comm | None = None,
    ) -> Self:
        """Return an application mapping using two orderings.

        The arrays ``app`` and ``petsc`` need NOT contain all the integers
        ``0`` to ``len(app)-1``, that is there CAN be "holes" in the indices.
        Use ``createBasic`` if they do not have holes for better performance.

        Parameters
        ----------
        app
            The application ordering.
        petsc
            Another ordering. May be `None` to indicate the identity ordering.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        createBasic, petsc.AOCreateMappingIS, petsc.AOCreateMapping

        """
        cdef PetscIS isapp = NULL, ispetsc = NULL
        cdef PetscInt napp = 0, *idxapp = NULL,
        cdef PetscInt npetsc = 0, *idxpetsc = NULL
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscAO newao = NULL
        if isinstance(app, IS):
            isapp = (<IS>app).iset
            if petsc is not None:
                ispetsc = (<IS?>petsc).iset
            CHKERR( AOCreateMappingIS(isapp, ispetsc, &newao) )
        else:
            app = iarray_i(app, &napp, &idxapp)
            if petsc is not None:
                petsc = iarray_i(petsc, &npetsc, &idxpetsc)
                assert napp == npetsc, "incompatible array sizes"
            CHKERR( AOCreateMapping(ccomm, napp, idxapp, idxpetsc, &newao) )
        PetscCLEAR(self.obj); self.ao = newao
        return self

    def getType(self) -> str:
        """Return the application ordering type.

        Not collective.

        See Also
        --------
        petsc.AOGetType

        """
        cdef PetscAOType cval = NULL
        CHKERR( AOGetType(self.ao, &cval) )
        return bytes2str(cval)

    def app2petsc(self, indices: Sequence[int] | IS) -> Sequence[int] | IS:
        """Map an application-defined ordering to the PETSc ordering.

        Collective.

        Any integers in ``indices`` that are negative are left unchanged. This
        allows one to convert, for example, neighbor lists that use negative
        entries to indicate nonexistent neighbors due to boundary conditions,
        etc.

        Integers that are out of range are mapped to -1.

        If ``IS`` is used, it cannot be of type stride or block.

        Parameters
        ----------
        indices
            The indices; to be replaced with their mapped values.

        See Also
        --------
        petsc2app, petsc.AOApplicationToPetscIS, petsc.AOApplicationToPetsc

        """
        cdef PetscIS iset = NULL
        cdef PetscInt nidx = 0, *idx = NULL
        if isinstance(indices, IS):
            iset = (<IS>indices).iset
            CHKERR( AOApplicationToPetscIS(self.ao, iset) )
        else:
            indices = oarray_i(indices, &nidx, &idx)
            CHKERR( AOApplicationToPetsc(self.ao, nidx, idx) )
        return indices

    def petsc2app(self, indices: Sequence[int] | IS) -> Sequence[int] | IS:
        """Map a PETSc ordering to the application-defined ordering.

        Collective.

        Any integers in ``indices`` that are negative are left unchanged. This
        allows one to convert, for example, neighbor lists that use negative
        entries to indicate nonexistent neighbors due to boundary conditions,
        etc.

        Integers that are out of range are mapped to -1.

        If ``IS`` is used, it cannot be of type stride or block.

        Parameters
        ----------
        indices
            The indices; to be replaced with their mapped values.

        See Also
        --------
        app2petsc, petsc.AOPetscToApplicationIS, petsc.AOPetscToApplication

        """
        cdef PetscIS iset = NULL
        cdef PetscInt nidx = 0, *idx = NULL
        if isinstance(indices, IS):
            iset = (<IS>indices).iset
            CHKERR( AOPetscToApplicationIS(self.ao, iset) )
        else:
            indices = oarray_i(indices, &nidx, &idx)
            CHKERR( AOPetscToApplication(self.ao, nidx, idx) )
        return indices

# --------------------------------------------------------------------

del AOType

# --------------------------------------------------------------------
