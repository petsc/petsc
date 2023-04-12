# --------------------------------------------------------------------

class DSType(object):
    BASIC = S_(PETSCDSBASIC)

# --------------------------------------------------------------------

cdef class DS(Object):
    """Discrete System object."""

    Type = DSType

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.ds
        self.ds  = NULL

    def view(self, Viewer viewer=None) -> None:
        """View a discrete system.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` to display the system.

        See Also
        --------
        petsc.PetscDSView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscDSView(self.ds, vwr) )

    def destroy(self) -> Self:
        """Destroy the discrete system.

        Collective.

        See Also
        --------
        create, petsc.PetscDSDestroy

        """
        CHKERR( PetscDSDestroy(&self.ds) )
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create an empty DS.

        Collective.

        The type can then be set with `setType`.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        setType, destroy, petsc.PetscDSCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDS newds = NULL
        CHKERR( PetscDSCreate(ccomm, &newds) )
        PetscCLEAR(self.obj); self.ds = newds
        return self

    def setType(self, ds_type: Type | str) -> None:
        """Build a particular type of a discrete system.

        Collective.

        Parameters
        ----------
        ds_type
            The type of the discrete system.

        See Also
        --------
        getType, petsc.PetscDSSetType

        """
        cdef PetscDSType cval = NULL
        ds_type = str2bytes(ds_type, &cval)
        CHKERR( PetscDSSetType(self.ds, cval) )

    def getType(self) -> str:
        """Return the type of the discrete system.

        Not collective.

        See Also
        --------
        setType, petsc.PetscDSGetType

        """
        cdef PetscDSType cval = NULL
        CHKERR( PetscDSGetType(self.ds, &cval) )
        return bytes2str(cval)

    def setFromOptions(self) -> None:
        """Set parameters in a `DS` from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.PetscDSSetFromOptions

        """
        CHKERR( PetscDSSetFromOptions(self.ds) )

    def setUp(self) -> Self:
        """Construct data structures for the discrete system.

        Collective.

        See Also
        --------
        petsc.PetscDSSetUp

        """
        CHKERR( PetscDSSetUp(self.ds) )
        return self

    #

    def getSpatialDimension(self) -> int:
        """Return the spatial dimension of the DS.

        Not collective.

        The spatial dimension of the `DS` is the topological dimension of the
        discretizations.

        See Also
        --------
        petsc.PetscDSGetSpatialDimension

        """
        cdef PetscInt dim = 0
        CHKERR( PetscDSGetSpatialDimension(self.ds, &dim) )
        return toInt(dim)

    def getCoordinateDimension(self) -> int:
        """Return the coordinate dimension of the DS.

        Not collective.

        The coordinate dimension of the `DS` is the dimension of the space into
        which the discretiaztions are embedded.

        See Also
        --------
        petsc.PetscDSGetCoordinateDimension

        """
        cdef PetscInt dim = 0
        CHKERR( PetscDSGetCoordinateDimension(self.ds, &dim) )
        return toInt(dim)

    def getNumFields(self) -> int:
        """Return the number of fields in the DS.

        Not collective.

        See Also
        --------
        petsc.PetscDSGetNumFields

        """
        cdef PetscInt nf = 0
        CHKERR( PetscDSGetNumFields(self.ds, &nf) )
        return toInt(nf)

    def getFieldIndex(self, Object disc) -> int:
        """Return the index of the given field.

        Not collective.

        Parameters
        ----------
        disc
            The discretization object.

        See Also
        --------
        petsc.PetscDSGetFieldIndex

        """
        cdef PetscInt field = 0
        CHKERR( PetscDSGetFieldIndex(self.ds, disc.obj[0], &field) )
        return toInt(field)

    def getTotalDimensions(self) -> int:
        """Return the total size of the approximation space for this system.

        Not collective.

        See Also
        --------
        petsc.PetscDSGetTotalDimension

        """
        cdef PetscInt tdim = 0
        CHKERR( PetscDSGetTotalDimension(self.ds, &tdim) )
        return toInt(tdim)

    def getTotalComponents(self) -> int:
        """Return the total number of components in this system.

        Not collective.

        See Also
        --------
        petsc.PetscDSGetTotalComponents

        """
        cdef PetscInt tcmp = 0
        CHKERR( PetscDSGetTotalComponents(self.ds, &tcmp) )
        return toInt(tcmp)

    def getDimensions(self) -> ArrayInt:
        """Return the size of the space for each field on an evaluation point.

        Not collective.

        See Also
        --------
        petsc.PetscDSGetDimensions

        """
        cdef PetscInt nf = 0, *dims = NULL
        CHKERR( PetscDSGetNumFields(self.ds, &nf) )
        CHKERR( PetscDSGetDimensions(self.ds, &dims) )
        return array_i(nf, dims)

    def getComponents(self) -> ArrayInt:
        """Return the number of components for each field on an evaluation point.

        Not collective.

        See Also
        --------
        petsc.PetscDSGetComponents

        """
        cdef PetscInt nf = 0, *cmps = NULL
        CHKERR( PetscDSGetNumFields(self.ds, &nf) )
        CHKERR( PetscDSGetComponents(self.ds, &cmps) )
        return array_i(nf, cmps)

    def setDiscretisation(self, f: int, disc: Object) -> None:
        """Set the discretization object for the given field.

        Not collective.

        Parameters
        ----------
        f
            The field number.
        disc
            The discretization object.

        See Also
        --------
        petsc.PetscDSSetDiscretization

        """
        cdef PetscInt cf = asInt(f)
        cdef FE fe = disc
        CHKERR( PetscDSSetDiscretization(self.ds, cf, <PetscObject> fe.fe) )



# --------------------------------------------------------------------

del DSType

# --------------------------------------------------------------------
