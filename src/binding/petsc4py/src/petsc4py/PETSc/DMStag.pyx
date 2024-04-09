# --------------------------------------------------------------------

class DMStagStencilType(object):
    """Stencil types."""
    STAR = DMSTAG_STENCIL_STAR
    BOX  = DMSTAG_STENCIL_BOX
    NONE = DMSTAG_STENCIL_NONE


class DMStagStencilLocation(object):
    """Stencil location types."""
    NULLLOC          = DMSTAG_NULL_LOCATION
    BACK_DOWN_LEFT   = DMSTAG_BACK_DOWN_LEFT
    BACK_DOWN        = DMSTAG_BACK_DOWN
    BACK_DOWN_RIGHT  = DMSTAG_BACK_DOWN_RIGHT
    BACK_LEFT        = DMSTAG_BACK_LEFT
    BACK             = DMSTAG_BACK
    BACK_RIGHT       = DMSTAG_BACK_RIGHT
    BACK_UP_LEFT     = DMSTAG_BACK_UP_LEFT
    BACK_UP          = DMSTAG_BACK_UP
    BACK_UP_RIGHT    = DMSTAG_BACK_UP_RIGHT
    DOWN_LEFT        = DMSTAG_DOWN_LEFT
    DOWN             = DMSTAG_DOWN
    DOWN_RIGHT       = DMSTAG_DOWN_RIGHT
    LEFT             = DMSTAG_LEFT
    ELEMENT          = DMSTAG_ELEMENT
    RIGHT            = DMSTAG_RIGHT
    UP_LEFT          = DMSTAG_UP_LEFT
    UP               = DMSTAG_UP
    UP_RIGHT         = DMSTAG_UP_RIGHT
    FRONT_DOWN_LEFT  = DMSTAG_FRONT_DOWN_LEFT
    FRONT_DOWN       = DMSTAG_FRONT_DOWN
    FRONT_DOWN_RIGHT = DMSTAG_FRONT_DOWN_RIGHT
    FRONT_LEFT       = DMSTAG_FRONT_LEFT
    FRONT            = DMSTAG_FRONT
    FRONT_RIGHT      = DMSTAG_FRONT_RIGHT
    FRONT_UP_LEFT    = DMSTAG_FRONT_UP_LEFT
    FRONT_UP         = DMSTAG_FRONT_UP
    FRONT_UP_RIGHT   = DMSTAG_FRONT_UP_RIGHT

# --------------------------------------------------------------------


cdef class DMStag(DM):
    """A DM object representing a "staggered grid" or a structured cell complex."""

    StencilType       = DMStagStencilType
    StencilLocation   = DMStagStencilLocation

    def create(
        self,
        dim: int,
        dofs: tuple[int, ...] | None = None,
        sizes: tuple[int, ...] | None = None,
        boundary_types: tuple[DM.BoundaryType | int | str | bool, ...] | None = None,
        stencil_type: StencilType | None = None,
        stencil_width: int | None = None,
        proc_sizes: tuple[int, ...] | None = None,
        ownership_ranges: tuple[Sequence[int], ...] | None = None,
        comm: Comm | None = None,
        setUp: bool | None = False) -> Self:
        """Create a DMDA object.

        Collective.

        Creates an object to manage data living on the elements and vertices /
        the elements, faces, and vertices / the elements, faces, edges, and
        vertices of a parallelized regular 1D / 2D / 3D grid.

        Parameters
        ----------
        dim
            The number of dimensions.
        dofs
            The number of degrees of freedom per vertex, element (1D); vertex,
            face, element (2D); or vertex, edge, face, element (3D).
        sizes
            The number of elements in each dimension.
        boundary_types
            The boundary types.
        stencil_type
            The ghost/halo stencil type.
        stencil_width
            The width of the ghost/halo region.
        proc_sizes
            The number of processes in x, y, z dimensions.
        ownership_ranges
            Local x, y, z element counts, of length equal to ``proc_sizes``,
            summing to ``sizes``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.
        setUp
            Whether to call the setup routine after creating the object.

        See Also
        --------
        petsc.DMStagCreate1d, petsc.DMStagCreate2d, petsc.DMStagCreate3d
        petsc.DMSetUp

        """
        # ndim
        cdef PetscInt ndim = asInt(dim)

        # sizes
        cdef object gsizes = sizes
        cdef PetscInt nsizes=PETSC_DECIDE, M=1, N=1, P=1
        if sizes is not None:
            nsizes = asStagDims(gsizes, &M, &N, &P)
            assert(nsizes==ndim)

        # dofs
        cdef object cdofs = dofs
        cdef PetscInt ndofs=PETSC_DECIDE, dof0=1, dof1=0, dof2=0, dof3=0
        if dofs is not None:
            ndofs = asDofs(cdofs, &dof0, &dof1, &dof2, &dof3)
            assert(ndofs==ndim+1)

        # boundary types
        cdef PetscDMBoundaryType btx = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType bty = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType btz = DM_BOUNDARY_NONE
        asBoundary(boundary_types, &btx, &bty, &btz)

        # stencil
        cdef PetscInt swidth = 0
        if stencil_width is not None:
            swidth = asInt(stencil_width)
        cdef PetscDMStagStencilType stype = DMSTAG_STENCIL_NONE
        if stencil_type is not None:
            stype = asStagStencil(stencil_type)

        # comm
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)

        # proc sizes
        cdef object psizes = proc_sizes
        cdef PetscInt nprocs=PETSC_DECIDE, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        if proc_sizes is not None:
            nprocs = asStagDims(psizes, &m, &n, &p)
            assert(nprocs==ndim)

        # ownership ranges
        cdef PetscInt *lx = NULL, *ly = NULL, *lz = NULL
        if ownership_ranges is not None:
            nranges = asStagOwnershipRanges(ownership_ranges, ndim, &m, &n, &p, &lx, &ly, &lz)
            assert(nranges==ndim)

        # create
        cdef PetscDM newda = NULL
        if dim == 1:
            CHKERR(DMStagCreate1d(ccomm, btx, M, dof0, dof1, stype, swidth, lx, &newda))
        if dim == 2:
            CHKERR(DMStagCreate2d(ccomm, btx, bty, M, N, m, n, dof0, dof1, dof2, stype, swidth, lx, ly, &newda))
        if dim == 3:
            CHKERR(DMStagCreate3d(ccomm, btx, bty, btz, M, N, P, m, n, p, dof0, dof1, dof2, dof3, stype, swidth, lx, ly, lz, &newda))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newda
        if setUp:
            CHKERR(DMSetUp(self.dm))
        return self

    # Setters

    def setStencilWidth(self, swidth: int) -> None:
        """Set elementwise stencil width.

        Logically collective.

        The width value is not used when `StencilType.NONE` is specified.

        Parameters
        ----------
        swidth
            Stencil/halo/ghost width in elements.

        See Also
        --------
        petsc.DMStagSetStencilWidth

        """
        cdef PetscInt sw = asInt(swidth)
        CHKERR(DMStagSetStencilWidth(self.dm, sw))

    def setStencilType(self, stenciltype: StencilType | str) -> None:
        """Set elementwise ghost/halo stencil type.

        Logically collective.

        Parameters
        ----------
        stenciltype
            The elementwise ghost stencil type.

        See Also
        --------
        getStencilType, petsc.DMStagSetStencilType

        """
        cdef PetscDMStagStencilType stype = asStagStencil(stenciltype)
        CHKERR(DMStagSetStencilType(self.dm, stype))

    def setBoundaryTypes(
        self,
        boundary_types: tuple[DM.BoundaryType | int | str | bool, ...]) -> None:
        """Set the boundary types.

        Logically collective.

        Parameters
        ----------
        boundary_types
            Boundary types for one/two/three dimensions.

        See Also
        --------
        getBoundaryTypes, petsc.DMStagSetBoundaryTypes

        """
        cdef PetscDMBoundaryType btx = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType bty = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType btz = DM_BOUNDARY_NONE
        asBoundary(boundary_types, &btx, &bty, &btz)
        CHKERR(DMStagSetBoundaryTypes(self.dm, btx, bty, btz))

    def setDof(self, dofs: tuple[int, ...]) -> None:
        """Set DOFs/stratum.

        Logically collective.

        Parameters
        ----------
        dofs
            The number of points per 0-cell (vertex/node), 1-cell (element in
            1D, edge in 2D and 3D), 2-cell (element in 2D, face in 3D), or
            3-cell (element in 3D).

        See Also
        --------
        petsc.DMStagSetDOF

        """
        cdef tuple gdofs = tuple(dofs)
        cdef PetscInt dof0=1, dof1=0, dof2=0, dof3=0
        asDofs(gdofs, &dof0, &dof1, &dof2, &dof3)
        CHKERR(DMStagSetDOF(self.dm, dof0, dof1, dof2, dof3))

    def setGlobalSizes(self, sizes: tuple[int, ...]) -> None:
        """Set global element counts in each dimension.

        Logically collective.

        Parameters
        ----------
        sizes
            Global elementwise size in the one/two/three dimensions.

        See Also
        --------
        petsc.DMStagSetGlobalSizes

        """
        cdef tuple gsizes = tuple(sizes)
        cdef PetscInt M=1, N=1, P=1
        asStagDims(gsizes, &M, &N, &P)
        CHKERR(DMStagSetGlobalSizes(self.dm, M, N, P))

    def setProcSizes(self, sizes: tuple[int, ...]) -> None:
        """Set the number of processes in each dimension in the global process grid.

        Logically collective.

        Parameters
        ----------
        sizes
            Number of processes in one/two/three dimensions.

        See Also
        --------
        petsc.DMStagSetNumRanks

        """
        cdef tuple psizes = tuple(sizes)
        cdef PetscInt m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        asStagDims(psizes, &m, &n, &p)
        CHKERR(DMStagSetNumRanks(self.dm, m, n, p))

    def setOwnershipRanges(self, ranges: tuple[Sequence[int], ...]) -> None:
        """Set elements per process in each dimension.

        Logically collective.

        Parameters
        ----------
        ranges
            Element counts for each process in one/two/three dimensions.

        See Also
        --------
        getOwnershipRanges, petsc.DMStagSetOwnershipRanges

        """
        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        cdef PetscInt *lx = NULL, *ly = NULL, *lz = NULL
        CHKERR(DMGetDimension(self.dm, &dim))
        CHKERR(DMStagGetNumRanks(self.dm, &m, &n, &p))
        asStagOwnershipRanges(ranges, dim, &m, &n, &p, &lx, &ly, &lz)
        CHKERR(DMStagSetOwnershipRanges(self.dm, lx, ly, lz))

    # Getters

    def getDim(self) -> int:
        """Return the number of dimensions.

        Not collective.

        """
        return self.getDimension()

    def getEntriesPerElement(self) -> int:
        """Return the number of entries per element in the local representation.

        Not collective.

        This is the natural block size for most local operations.

        See Also
        --------
        petsc.DMStagGetEntriesPerElement

        """
        cdef PetscInt epe=0
        CHKERR(DMStagGetEntriesPerElement(self.dm, &epe))
        return toInt(epe)

    def getStencilWidth(self) -> int:
        """Return elementwise stencil width.

        Not collective.

        See Also
        --------
        petsc.DMStagGetStencilWidth

        """
        cdef PetscInt swidth=0
        CHKERR(DMStagGetStencilWidth(self.dm, &swidth))
        return toInt(swidth)

    def getDof(self) -> tuple[int, ...]:
        """Get number of DOFs associated with each stratum of the grid.

        Not collective.

        See Also
        --------
        petsc.DMStagGetDOF

        """
        cdef PetscInt dim=0, dof0=0, dof1=0, dof2=0, dof3=0
        CHKERR(DMStagGetDOF(self.dm, &dof0, &dof1, &dof2, &dof3))
        CHKERR(DMGetDimension(self.dm, &dim))
        return toDofs(dim+1, dof0, dof1, dof2, dof3)

    def getCorners(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Return starting element index, width and number of partial elements.

        Not collective.

        The returned value is calculated excluding ghost points.

        The number of extra partial elements is either ``1`` or ``0``. The
        value is ``1`` on right, top, and front non-periodic domain
        ("physical") boundaries, in the x, y, and z dimensions respectively,
        and otherwise ``0``.

        See Also
        --------
        getGhostCorners, petsc.DMStagGetCorners, petsc.DMGetDimension

        """
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0, nExtrax=0, nExtray=0, nExtraz=0
        CHKERR(DMGetDimension(self.dm, &dim))
        CHKERR(DMStagGetCorners(self.dm, &x, &y, &z, &m, &n, &p, &nExtrax, &nExtray, &nExtraz))
        return (asInt(x), asInt(y), asInt(z))[:<Py_ssize_t>dim], (asInt(m), asInt(n), asInt(p))[:<Py_ssize_t>dim], (asInt(nExtrax), asInt(nExtray), asInt(nExtraz))[:<Py_ssize_t>dim]

    def getGhostCorners(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return starting element index and width of local region.

        Not collective.

        See Also
        --------
        getCorners, petsc.DMStagGetGhostCorners

        """
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR(DMGetDimension(self.dm, &dim))
        CHKERR(DMStagGetGhostCorners(self.dm, &x, &y, &z, &m, &n, &p))
        return (asInt(x), asInt(y), asInt(z))[:<Py_ssize_t>dim], (asInt(m), asInt(n), asInt(p))[:<Py_ssize_t>dim]

    def getLocalSizes(self) -> tuple[int, ...]:
        """Return local elementwise sizes in each dimension.

        Not collective.

        The returned value is calculated excluding ghost points.

        See Also
        --------
        getGlobalSizes, petsc.DMStagGetLocalSizes

        """
        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        CHKERR(DMGetDimension(self.dm, &dim))
        CHKERR(DMStagGetLocalSizes(self.dm, &m, &n, &p))
        return toStagDims(dim, m, n, p)

    def getGlobalSizes(self) -> tuple[int, ...]:
        """Return global element counts in each dimension.

        Not collective.

        See Also
        --------
        getLocalSizes, petsc.DMStagGetGlobalSizes

        """
        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        CHKERR(DMGetDimension(self.dm, &dim))
        CHKERR(DMStagGetGlobalSizes(self.dm, &m, &n, &p))
        return toStagDims(dim, m, n, p)

    def getProcSizes(self) -> tuple[int, ...]:
        """Return number of processes in each dimension.

        Not collective.

        See Also
        --------
        petsc.DMStagGetNumRanks

        """
        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        CHKERR(DMGetDimension(self.dm, &dim))
        CHKERR(DMStagGetNumRanks(self.dm, &m, &n, &p))
        return toStagDims(dim, m, n, p)

    def getStencilType(self) -> str:
        """Return elementwise ghost/halo stencil type.

        Not collective.

        See Also
        --------
        setStencilType, petsc.DMStagGetStencilType

        """
        cdef PetscDMStagStencilType stype = DMSTAG_STENCIL_BOX
        CHKERR(DMStagGetStencilType(self.dm, &stype))
        return toStagStencil(stype)

    def getOwnershipRanges(self) -> tuple[Sequence[int], ...]:
        """Return elements per process in each dimension.

        Not collective.

        See Also
        --------
        setOwnershipRanges, petsc.DMStagGetOwnershipRanges

        """
        cdef PetscInt dim=0, m=0, n=0, p=0
        cdef const PetscInt *lx = NULL, *ly = NULL, *lz = NULL
        CHKERR(DMGetDimension(self.dm, &dim))
        CHKERR(DMStagGetNumRanks(self.dm, &m, &n, &p))
        CHKERR(DMStagGetOwnershipRanges(self.dm, &lx, &ly, &lz))
        return toStagOwnershipRanges(dim, m, n, p, lx, ly, lz)

    def getBoundaryTypes(self) -> tuple[str, ...]:
        """Return boundary types in each dimension.

        Not collective.

        See Also
        --------
        setBoundaryTypes, petsc.DMStagGetBoundaryTypes

        """
        cdef PetscInt dim=0
        cdef PetscDMBoundaryType btx = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType bty = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType btz = DM_BOUNDARY_NONE
        CHKERR(DMGetDimension(self.dm, &dim))
        CHKERR(DMStagGetBoundaryTypes(self.dm, &btx, &bty, &btz))
        return toStagBoundaryTypes(dim, btx, bty, btz)

    def getIsFirstRank(self) -> tuple[int, ...]:
        """Return whether this process is first in each dimension in the process grid.

        Not collective.

        See Also
        --------
        petsc.DMStagGetIsFirstRank

        """
        cdef PetscBool rank0=PETSC_FALSE, rank1=PETSC_FALSE, rank2=PETSC_FALSE
        cdef PetscInt dim=0
        CHKERR(DMGetDimension(self.dm, &dim))
        CHKERR(DMStagGetIsFirstRank(self.dm, &rank0, &rank1, &rank2))
        return toStagDims(dim, rank0, rank1, rank2)

    def getIsLastRank(self) -> tuple[int, ...]:
        """Return whether this process is last in each dimension in the process grid.

        Not collective.

        See Also
        --------
        petsc.DMStagGetIsLastRank

        """
        cdef PetscBool rank0=PETSC_FALSE, rank1=PETSC_FALSE, rank2=PETSC_FALSE
        cdef PetscInt dim=0
        CHKERR(DMGetDimension(self.dm, &dim))
        CHKERR(DMStagGetIsLastRank(self.dm, &rank0, &rank1, &rank2))
        return toStagDims(dim, rank0, rank1, rank2)

    # Coordinate-related functions

    def setUniformCoordinatesExplicit(
        self,
        xmin: float = 0,
        xmax: float = 1,
        ymin: float = 0,
        ymax: float = 1,
        zmin: float = 0,
        zmax: float = 1) -> None:
        """Set coordinates to be a uniform grid, storing all values.

        Collective.

        Parameters
        ----------
        xmin
            The minimum global coordinate value in the x dimension.
        xmax
            The maximum global coordinate value in the x dimension.
        ymin
            The minimum global coordinate value in the y dimension.
        ymax
            The maximum global coordinate value in the y dimension.
        zmin
            The minimum global coordinate value in the z dimension.
        zmax
            The maximum global coordinate value in the z dimension.

        See Also
        --------
        setUniformCoordinatesProduct, setUniformCoordinates
        petsc.DMStagSetUniformCoordinatesExplicit

        """
        cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
        cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
        cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
        CHKERR(DMStagSetUniformCoordinatesExplicit(self.dm, _xmin, _xmax, _ymin, _ymax, _zmin, _zmax))

    def setUniformCoordinatesProduct(
        self,
        xmin: float = 0,
        xmax: float = 1,
        ymin: float = 0,
        ymax: float = 1,
        zmin: float = 0,
        zmax: float = 1) -> None:
        """Create uniform coordinates, as a product of 1D arrays.

        Collective.

        The per-dimension 1-dimensional `DMStag` objects that comprise the
        product always have active 0-cells (vertices, element boundaries) and
        1-cells (element centers).

        Parameters
        ----------
        xmin
            The minimum global coordinate value in the x dimension.
        xmax
            The maximum global coordinate value in the x dimension.
        ymin
            The minimum global coordinate value in the y dimension.
        ymax
            The maximum global coordinate value in the y dimension.
        zmin
            The minimum global coordinate value in the z dimension.
        zmax
            The maximum global coordinate value in the z dimension.

        See Also
        --------
        setUniformCoordinatesExplicit, setUniformCoordinates
        petsc.DMStagSetUniformCoordinatesProduct

        """
        cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
        cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
        cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
        CHKERR(DMStagSetUniformCoordinatesProduct(self.dm, _xmin, _xmax, _ymin, _ymax, _zmin, _zmax))

    def setUniformCoordinates(
        self,
        xmin: float = 0,
        xmax: float = 1,
        ymin: float = 0,
        ymax: float = 1,
        zmin: float = 0,
        zmax: float = 1) -> None:
        """Set the coordinates to be a uniform grid..

        Collective.

        Local coordinates are populated, linearly extrapolated to ghost cells,
        including those outside the physical domain. This is also done in case
        of periodic boundaries, meaning that the same global point may have
        different coordinates in different local representations, which are
        equivalent assuming a periodicity implied by the arguments to this
        function, i.e., two points are equivalent if their difference is a
        multiple of ``xmax-xmin`` in the x dimension, ``ymax-ymin`` in the y
        dimension, and ``zmax-zmin`` in the z dimension.

        Parameters
        ----------
        xmin
            The minimum global coordinate value in the x dimension.
        xmax
            The maximum global coordinate value in the x dimension.
        ymin
            The minimum global coordinate value in the y dimension.
        ymax
            The maximum global coordinate value in the y dimension.
        zmin
            The minimum global coordinate value in the z dimension.
        zmax
            The maximum global coordinate value in the z dimension.

        See Also
        --------
        setUniformCoordinatesExplicit, setUniformCoordinatesProduct
        petsc.DMStagSetUniformCoordinates

        """
        cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
        cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
        cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
        CHKERR(DMStagSetUniformCoordinates(self.dm, _xmin, _xmax, _ymin, _ymax, _zmin, _zmax))

    def setCoordinateDMType(self, dmtype: DM.Type) -> None:
        """Set the type to store coordinates.

        Logically collective.

        Parameters
        ----------
        dmtype
            The type to store coordinates.

        See Also
        --------
        petsc.DMStagSetCoordinateDMType

        """
        cdef PetscDMType cval = NULL
        dmtype = str2bytes(dmtype, &cval)
        CHKERR(DMStagSetCoordinateDMType(self.dm, cval))

    # Location slot related functions

    def getLocationSlot(self, loc: StencilLocation, c: int) -> int:
        """Return index to use in accessing raw local arrays.

        Not collective.

        Parameters
        ----------
        loc
            Location relative to an element.
        c
            Component.

        See Also
        --------
        petsc.DMStagGetLocationSlot

        """
        cdef PetscInt slot=0
        cdef PetscInt comp=asInt(c)
        cdef PetscDMStagStencilLocation sloc = asStagStencilLocation(loc)
        CHKERR(DMStagGetLocationSlot(self.dm, sloc, comp, &slot))
        return toInt(slot)

    def getProductCoordinateLocationSlot(self, loc: StencilLocation) -> None:
        """Return slot for use with local product coordinate arrays.

        Not collective.

        Parameters
        ----------
        loc
            The grid location.

        See Also
        --------
        petsc.DMStagGetProductCoordinateLocationSlot

        """
        cdef PetscInt slot=0
        cdef PetscDMStagStencilLocation sloc = asStagStencilLocation(loc)
        CHKERR(DMStagGetProductCoordinateLocationSlot(self.dm, sloc, &slot))
        return toInt(slot)

    def getLocationDof(self, loc: StencilLocation) -> int:
        """Return number of DOFs associated with a given point on the grid.

        Not collective.

        Parameters
        ----------
        loc
            The grid point.

        See Also
        --------
        petsc.DMStagGetLocationDOF

        """
        cdef PetscInt dof=0
        cdef PetscDMStagStencilLocation sloc = asStagStencilLocation(loc)
        CHKERR(DMStagGetLocationDOF(self.dm, sloc, &dof))
        return toInt(dof)

    # Random other functions

    def migrateVec(self, Vec vec, DM dmTo, Vec vecTo) -> None:
        """Transfer a vector between two ``DMStag`` objects.

        Collective.

        Currently only implemented to migrate global vectors to global vectors.

        Parameters
        ----------
        vec
            The source vector.
        dmTo
            The compatible destination object.
        vecTo
            The destination vector.

        See Also
        --------
        petsc.DMStagMigrateVec

        """
        CHKERR(DMStagMigrateVec(self.dm, vec.vec, dmTo.dm, vecTo.vec))

    def createCompatibleDMStag(self, dofs: tuple[int, ...]) -> DM:
        """Create a compatible ``DMStag`` with different DOFs/stratum.

        Collective.

        Parameters
        ----------
        dofs
            The number of DOFs on the strata in the new `DMStag`.

        See Also
        --------
        petsc.DMStagCreateCompatibleDMStag

        """
        cdef tuple gdofs = tuple(dofs)
        cdef PetscInt dof0=1, dof1=0, dof2=0, dof3=0
        asDofs(gdofs, &dof0, &dof1, &dof2, &dof3)
        cdef PetscDM newda = NULL
        CHKERR(DMStagCreateCompatibleDMStag(self.dm, dof0, dof1, dof2, dof3, &newda))
        cdef DM newdm = type(self)()
        CHKERR(PetscCLEAR(newdm.obj)); newdm.dm = newda
        return newdm

    def VecSplitToDMDA(
        self,
        Vec vec,
        loc: StencilLocation,
        c: int) -> tuple[DMDA, Vec]:
        """Return ``DMDA``, ``Vec`` from a subgrid of a ``DMStag``, its ``Vec``.

        Collective.

        If a ``c`` value of ``-k`` is provided, the first ``k`` DOFs for that
        position are extracted, padding with zero values if needed. If a
        non-negative value is provided, a single DOF is extracted.

        Parameters
        ----------
        vec
            The ``Vec`` object.
        loc
            Which subgrid to extract.
        c
            Which component to extract.

        See Also
        --------
        petsc.DMStagVecSplitToDMDA

        """
        cdef PetscInt pc = asInt(c)
        cdef PetscDMStagStencilLocation sloc = asStagStencilLocation(loc)
        cdef PetscDM pda = NULL
        cdef PetscVec pdavec = NULL
        CHKERR(DMStagVecSplitToDMDA(self.dm, vec.vec, sloc, pc, &pda, &pdavec))
        cdef DM da = DMDA()
        CHKERR(PetscCLEAR(da.obj)); da.dm = pda
        cdef Vec davec = Vec()
        CHKERR(PetscCLEAR(davec.obj)); davec.vec = pdavec
        return (da, davec)

    def getVecArray(self, Vec vec) -> None:
        """Not implemented."""
        raise NotImplementedError('getVecArray for DMStag not yet implemented in petsc4py')

    def get1dCoordinatecArrays(self) -> None:
        """Not implemented."""
        raise NotImplementedError('get1dCoordinatecArrays for DMStag not yet implemented in petsc4py')

    property dim:
        """The dimension."""
        def __get__(self) -> int:
            return self.getDim()

    property dofs:
        """The number of DOFs associated with each stratum of the grid."""
        def __get__(self) -> tuple[int, ...]:
            return self.getDof()

    property entries_per_element:
        """The number of entries per element in the local representation."""
        def __get__(self) -> int:
            return self.getEntriesPerElement()

    property global_sizes:
        """Global element counts in each dimension."""
        def __get__(self) -> tuple[int, ...]:
            return self.getGlobalSizes()

    property local_sizes:
        """Local elementwise sizes in each dimension."""
        def __get__(self) -> tuple[int, ...]:
            return self.getLocalSizes()

    property proc_sizes:
        """The number of processes in each dimension in the global decomposition."""
        def __get__(self) -> tuple[int, ...]:
            return self.getProcSizes()

    property boundary_types:
        """Boundary types in each dimension."""
        def __get__(self) -> tuple[str, ...]:
            return self.getBoundaryTypes()

    property stencil_type:
        """Stencil type."""
        def __get__(self) -> str:
            return self.getStencilType()

    property stencil_width:
        """Elementwise stencil width."""
        def __get__(self) -> int:
            return self.getStencilWidth()

    property corners:
        """The lower left corner and size of local region in each dimension."""
        def __get__(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
            return self.getCorners()

    property ghost_corners:
        """The lower left corner and size of local region in each dimension."""
        def __get__(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
            return self.getGhostCorners()


# --------------------------------------------------------------------

del DMStagStencilType
del DMStagStencilLocation

# --------------------------------------------------------------------
