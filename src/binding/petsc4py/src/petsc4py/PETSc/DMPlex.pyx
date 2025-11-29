# --------------------------------------------------------------------

cdef class DMPlex(DM):
    """Encapsulate an unstructured mesh.

    DMPlex encapsulates both topology and geometry.
    It is capable of parallel refinement and coarsening (using Pragmatic or ParMmg)
    and parallel redistribution for load balancing.

    """

    #

    def create(self, comm: Comm | None = None) -> Self:
        """Create a `DMPlex` object, which encapsulates an unstructured mesh.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        DM, DMPlex, DM.create, DM.setType, petsc.DMPlexCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR(DMPlexCreate(ccomm, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm
        return self

    def createFromCellList(self, dim: int, cells: Sequence[int], coords: Sequence[float], interpolate: bool | None = True, comm: Comm | None = None) -> Self:
        """Create a `DMPlex` from a list of vertices for each cell on process 0.

        Collective.

        Parameters
        ----------
        dim
            The topological dimension of the mesh.
        cells
            An array of number of cells times number of vertices on each cell.
        coords
            An array of number of vertices times spatial dimension for coordinates.
        interpolate
            Flag to interpolate the mesh.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.interpolate,
        petsc.DMPlexCreateFromCellListPetsc

        """
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef PetscInt  cdim = asInt(dim)
        cdef PetscInt  numCells = 0
        cdef PetscInt  numCorners = 0
        cdef PetscInt  *cellVertices = NULL
        cdef PetscInt  numVertices = 0
        cdef PetscInt  spaceDim= 0
        cdef PetscReal *vertexCoords = NULL
        cdef int npy_flags = NPY_ARRAY_ALIGNED|NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_CARRAY
        cells  = PyArray_FROM_OTF(cells,  NPY_PETSC_INT,  npy_flags)
        coords = PyArray_FROM_OTF(coords, NPY_PETSC_REAL, npy_flags)
        if PyArray_NDIM(cells) != 2: raise ValueError(
                ("cell indices must have two dimensions: "
                 "cells.ndim=%d") % (PyArray_NDIM(cells)))
        if PyArray_NDIM(coords) != 2: raise ValueError(
                ("coords vertices must have two dimensions: "
                 "coords.ndim=%d") % (PyArray_NDIM(coords)))
        numCells     = <PetscInt> PyArray_DIM(cells, 0)
        numCorners   = <PetscInt> PyArray_DIM(cells, 1)
        numVertices  = <PetscInt> PyArray_DIM(coords, 0)
        spaceDim     = <PetscInt> PyArray_DIM(coords, 1)
        cellVertices = <PetscInt*> PyArray_DATA(cells)
        vertexCoords = <PetscReal*> PyArray_DATA(coords)
        CHKERR(DMPlexCreateFromCellListPetsc(ccomm, cdim, numCells, numVertices,
                                             numCorners, interp, cellVertices,
                                             spaceDim, vertexCoords, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm
        return self

    def createBoxMesh(self, faces: Sequence[int], lower: Sequence[float] | None = (0, 0, 0), upper: Sequence[float] | None = (1, 1, 1),
                      simplex: bool | None = True, periodic: Sequence | str | int | bool | None = False, interpolate: bool | None = True, localizationHeight: int | None = 0, sparseLocalize: bool | None = True, comm: Comm | None = None) -> Self:
        """Create a mesh on the tensor product of intervals.

        Collective.

        Parameters
        ----------
        faces
            Number of faces per dimension, or `None` for the default.
        lower
            The lower left corner.
        upper
            The upper right corner.
        simplex
            `True` for simplices, `False` for tensor cells.
        periodic
            The boundary type for the X, Y, Z direction,
            or `None` for `DM.BoundaryType.NONE`.
        interpolate
            Flag to create intermediate mesh entities (edges, faces).
        localizationHeight
            Flag to localize edges and faces in addition to cells;
            only significant for periodic meshes.
        sparseLocalize
            Flag to localize coordinates only for cells near the
            periodic boundary; only significant for periodic meshes.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        DM, DMPlex, DM.setFromOptions, DMPlex.createFromFile, DM.setType
        DM.create, petsc.DMPlexCreateBoxMesh

        """
        cdef Py_ssize_t i = 0
        cdef PetscInt dim = 0, *cfaces = NULL
        faces = iarray_i(faces, &dim, &cfaces)
        assert dim >= 1 and dim <= 3
        cdef PetscReal clower[3]
        clower[0] = clower[1] = clower[2] = 0
        for i from 0 <= i < dim: clower[i] = lower[i]
        cdef PetscReal cupper[3]
        cupper[0] = cupper[1] = cupper[2] = 1
        for i from 0 <= i < dim: cupper[i] = upper[i]
        cdef PetscDMBoundaryType btype[3]
        asBoundary(periodic, &btype[0], &btype[1], &btype[2])
        cdef PetscBool csimplex = simplex
        cdef PetscBool cinterp = interpolate
        cdef PetscInt  clocalizationHeight = asInt(localizationHeight)
        cdef PetscBool csparseLocalize = sparseLocalize
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM   newdm = NULL
        CHKERR(DMPlexCreateBoxMesh(ccomm, dim, csimplex, cfaces,
                                   clower, cupper, btype, cinterp, clocalizationHeight, csparseLocalize, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm
        return self

    def createBoxSurfaceMesh(self, faces: Sequence[int], lower: Sequence[float] | None = (0, 0, 0), upper: Sequence[float] | None = (1, 1, 1),
                             interpolate: bool | None = True, comm: Comm | None = None) -> Self:
        """Create a mesh on the surface of a box mesh using tensor cells.

        Collective.

        Parameters
        ----------
        faces
            Number of faces per dimension, or `None` for the default.
        lower
            The lower left corner.
        upper
            The upper right corner.
        interpolate
            Flag to create intermediate mesh pieces (edges, faces).
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        DM, DMPlex, DM.setFromOptions, DMPlex.createBoxMesh
        DMPlex.createFromFile, DM.setType, DM.create
        petsc.DMPlexCreateBoxSurfaceMesh

        """
        cdef Py_ssize_t i = 0
        cdef PetscInt dim = 0, *cfaces = NULL
        faces = iarray_i(faces, &dim, &cfaces)
        assert dim >= 1 and dim <= 3
        cdef PetscReal clower[3]
        clower[0] = clower[1] = clower[2] = 0
        for i from 0 <= i < dim: clower[i] = lower[i]
        cdef PetscReal cupper[3]
        cupper[0] = cupper[1] = cupper[2] = 1
        for i from 0 <= i < dim: cupper[i] = upper[i]
        cdef PetscBool cinterp = interpolate
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM   newdm = NULL
        CHKERR(DMPlexCreateBoxSurfaceMesh(ccomm, dim, cfaces, clower, cupper, cinterp, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm
        return self

    def createFromFile(self, filename: str, plexname: str | None = "unnamed", interpolate: bool | None = True, comm: Comm | None = None) -> Self:
        """Create `DMPlex` from a file.

        Collective.

        Parameters
        ----------
        filename
            A file name.
        plexname
            The name of the resulting `DMPlex`,
            also used for intra-datafile lookup by some formats.
        interpolate
            Flag to create intermediate mesh pieces (edges, faces).
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        DM, DMPlex, DMPlex.createFromCellList, DMPlex.create, Object.setName
        DM.view, DM.load, petsc_options, petsc.DMPlexCreateFromFile

        """
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef const char *cfile = NULL
        cdef const char *pname = NULL
        filename = str2bytes(filename, &cfile)
        plexname = str2bytes(plexname, &pname)
        CHKERR(DMPlexCreateFromFile(ccomm, cfile, pname, interp, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm
        return self

    def createCGNS(self, cgid: int, interpolate: bool | None = True, comm: Comm | None = None) -> Self:
        """Create a `DMPlex` mesh from a CGNS file.

        Collective.

        Parameters
        ----------
        cgid
            The CG id associated with a file and obtained using cg_open.
        interpolate
            Create faces and edges in the mesh.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.createCGNSFromFile
        DMPlex.createExodus, petsc.DMPlexCreateCGNS

        """
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef PetscInt  ccgid = asInt(cgid)
        CHKERR(DMPlexCreateCGNS(ccomm, ccgid, interp, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm
        return self

    def createCGNSFromFile(self, filename: str, interpolate: bool | None = True, comm: Comm | None = None) -> Self:
        """"Create a `DMPlex` mesh from a CGNS file.

        Collective.

        Parameters
        ----------
        filename
            The name of the CGNS file.
        interpolate
            Create faces and edges in the mesh.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.createCGNS, DMPlex.createExodus
        petsc.DMPlexCreateCGNS

        """
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef const char *cfile = NULL
        filename = str2bytes(filename, &cfile)
        CHKERR(DMPlexCreateCGNSFromFile(ccomm, cfile, interp, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm
        return self

    def createExodusFromFile(self, filename: str, interpolate: bool | None = True, comm: Comm | None = None) -> Self:
        """Create a `DMPlex` mesh from an ExodusII file.

        Collective.

        Parameters
        ----------
        filename
            The name of the ExodusII file.
        interpolate
            Create faces and edges in the mesh.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        DM, DMPlex, DM.create, DMPlex.createExodus
        petsc.DMPlexCreateExodusFromFile

        """
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef const char *cfile = NULL
        filename = str2bytes(filename, &cfile)
        CHKERR(DMPlexCreateExodusFromFile(ccomm, cfile, interp, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm
        return self

    def createExodus(self, exoid: int, interpolate: bool | None = True, comm: Comm | None = None) -> Self:
        """Create a `DMPlex` mesh from an ExodusII file ID.

        Collective.

        Parameters
        ----------
        exoid
            The ExodusII id associated with a file obtained using ``ex_open``.
        interpolate
            Create faces and edges in the mesh,
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        DM, DMPlex, DM.create, petsc.DMPlexCreateExodus

        """
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef PetscInt  cexoid = asInt(exoid)
        CHKERR(DMPlexCreateExodus(ccomm, <int> cexoid, interp, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm
        return self

    def createGmsh(self, Viewer viewer, interpolate: bool | None = True, comm: Comm | None = None) -> Self:
        """Create a `DMPlex` mesh from a Gmsh file viewer.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` associated with a Gmsh file.
        interpolate
            Create faces and edges in the mesh.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        Notes
        -----
        ``-dm_plex_gmsh_hybrid`` forces triangular prisms to use tensor order.\n
        ``-dm_plex_gmsh_periodic`` allows for reading Gmsh periodic section.\n
        ``-dm_plex_gmsh_highorder`` allows for generating high-order coordinates.\n
        ``-dm_plex_gmsh_project`` projects high-order coordinates to a different space,
        use the prefix ``-dm_plex_gmsh_project_`` to define the space.\n
        ``-dm_plex_gmsh_use_regions`` generates labels with region names.\n
        ``-dm_plex_gmsh_mark_vertices`` adds vertices to generated labels.\n
        ``-dm_plex_gmsh_multiple_tags`` allows multiple tags for default labels.\n
        ``-dm_plex_gmsh_spacedim <d>`` embedding space dimension.

        See Also
        --------
        DM, DMPlex, DM.create, petsc_options, petsc.DMPlexCreateGmsh

        """
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        CHKERR(DMPlexCreateGmsh(ccomm, viewer.vwr, interp, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm
        return self

    def createCoordinateSpace(self, degree: int, localized: bool, project: bool) -> None:
        """Create a finite element space for the coordinates.

        Collective.

        Parameters
        ----------
        degree
            The degree of the finite element.
        localized
            Flag to create a localized (DG) coordinate space.
        project
            Flag to project current coordinates into the space.

        See Also
        --------
        DM, DMPlex, petsc.DMPlexCreateCoordinateSpace

        """
        cdef PetscInt cdegree = asInt(degree)
        cdef PetscBool clocalized = localized
        cdef PetscBool cproject = project
        CHKERR(DMPlexCreateCoordinateSpace(self.dm, cdegree, clocalized, cproject))

    def createCohesiveSubmesh(self, hasLagrange: bool, value: int) -> DMPlex:
        """Extract the hypersurface defined by one face of the cohesive cells.

        Collective.

        Parameters
        ----------
        hasLagrange
            Flag indicating whether the mesh has Lagrange dofs in the cohesive cells.
        value
            A label value.

        See Also
        --------
        DM, DMPlex, petsc.DMPlexCreateCohesiveSubmesh

        """
        cdef PetscBool flag = hasLagrange
        cdef PetscInt cvalue = asInt(value)
        cdef DM subdm = DMPlex()
        CHKERR(DMPlexCreateCohesiveSubmesh(self.dm, flag, NULL, cvalue, &subdm.dm))
        return subdm

    def getChart(self) -> tuple[int, int]:
        """Return the interval for all mesh points [``pStart``, ``pEnd``).

        Not collective.

        Returns
        -------
        pStart : int
            The first mesh point.
        pEnd : int
            The upper bound for mesh points.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.setChart, petsc.DMPlexGetChart

        """
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR(DMPlexGetChart(self.dm, &pStart, &pEnd))
        return toInt(pStart), toInt(pEnd)

    def setChart(self, pStart: int, pEnd: int) -> None:
        """Set the interval for all mesh points [``pStart``, ``pEnd``).

        Not collective.

        Parameters
        ----------
        pStart
            The first mesh point.
        pEnd
            The upper bound for mesh points.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.getChart, petsc.DMPlexSetChart

        """
        cdef PetscInt cStart = asInt(pStart)
        cdef PetscInt cEnd   = asInt(pEnd)
        CHKERR(DMPlexSetChart(self.dm, cStart, cEnd))

    def getConeSize(self, p: int) -> int:
        """Return the number of in-edges for this point in the DAG.

        Not collective.

        Parameters
        ----------
        p
            The point, which must lie in the chart set with `DMPlex.setChart`.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.setConeSize, DMPlex.setChart
        petsc.DMPlexGetConeSize

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR(DMPlexGetChart(self.dm, &pStart, &pEnd))
        assert cp>=pStart and cp<pEnd
        cdef PetscInt csize = 0
        CHKERR(DMPlexGetConeSize(self.dm, cp, &csize))
        return toInt(csize)

    def setConeSize(self, p: int, size: int) -> None:
        """Set the number of in-edges for this point in the DAG.

        Not collective.

        Parameters
        ----------
        p
            The point, which must lie in the chart set with `DMPlex.setChart`.
        size
            The cone size for point ``p``.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.getConeSize, DMPlex.setChart
        petsc.DMPlexSetConeSize

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR(DMPlexGetChart(self.dm, &pStart, &pEnd))
        assert cp>=pStart and cp<pEnd
        cdef PetscInt csize = asInt(size)
        CHKERR(DMPlexSetConeSize(self.dm, cp, csize))

    def getCone(self, p: int) -> ArrayInt:
        """Return the points on the in-edges for this point in the DAG.

        Not collective.

        Parameters
        ----------
        p
            The point, which must lie in the chart set with `DMPlex.setChart`.

        See Also
        --------
        DM, DMPlex, DMPlex.getConeSize, DMPlex.setCone, DMPlex.setChart
        petsc.DMPlexGetCone

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR(DMPlexGetChart(self.dm, &pStart, &pEnd))
        assert cp>=pStart and cp<pEnd
        cdef PetscInt        ncone = 0
        cdef const PetscInt *icone = NULL
        CHKERR(DMPlexGetConeSize(self.dm, cp, &ncone))
        CHKERR(DMPlexGetCone(self.dm, cp, &icone))
        return array_i(ncone, icone)

    def setCone(self, p: int, cone: Sequence[int], orientation: Sequence[int] | None = None) -> None:
        """Set the points on the in-edges for this point in the DAG.

        Not collective.

        Parameters
        ----------
        p
            The point, which must lie in the chart set with `DMPlex.setChart`.
        cone
            An array of points which are on the in-edges for point ``p``.
        orientation
            An array of orientations, defaults to `None`.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.getCone, DMPlex.setChart
        DMPlex.setConeSize, DM.setUp, DMPlex.setSupport
        DMPlex.setSupportSize, petsc.DMPlexSetCone

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR(DMPlexGetChart(self.dm, &pStart, &pEnd))
        assert cp>=pStart and cp<pEnd
        #
        cdef PetscInt  ncone = 0
        cdef PetscInt *icone = NULL
        cone = iarray_i(cone, &ncone, &icone)
        CHKERR(DMPlexSetConeSize(self.dm, cp, ncone))
        CHKERR(DMPlexSetCone(self.dm, cp, icone))
        #
        cdef PetscInt  norie = 0
        cdef PetscInt *iorie = NULL
        if orientation is not None:
            orientation = iarray_i(orientation, &norie, &iorie)
            assert norie == ncone
            CHKERR(DMPlexSetConeOrientation(self.dm, cp, iorie))

    def insertCone(self, p: int, conePos: int, conePoint: int) -> None:
        """DMPlexInsertCone - Insert a point into the in-edges for the point p in the DAG.

        Not collective.

        Parameters
        ----------
        p
            The point, which must lie in the chart set with `DMPlex.setChart`.
        conePos
            The local index in the cone where the point should be put.
        conePoint
            The mesh point to insert.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.getCone, DMPlex.setChart
        DMPlex.setConeSize, DM.setUp, petsc.DMPlexInsertCone

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt cconePos = asInt(conePos)
        cdef PetscInt cconePoint = asInt(conePoint)
        CHKERR(DMPlexInsertCone(self.dm, cp, cconePos, cconePoint))

    def insertConeOrientation(self, p: int, conePos: int, coneOrientation: int) -> None:
        """Insert a point orientation for the in-edge for the point p in the DAG.

        Not collective.

        Parameters
        ----------
        p
            The point, which must lie in the chart set with `DMPlex.setChart`
        conePos
            The local index in the cone where the point should be put.
        coneOrientation
            The point orientation to insert.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.getCone, DMPlex.setChart
        DMPlex.setConeSize, DM.setUp, petsc.DMPlexInsertConeOrientation

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt cconePos = asInt(conePos)
        cdef PetscInt cconeOrientation = asInt(coneOrientation)
        CHKERR(DMPlexInsertConeOrientation(self.dm, cp, cconePos, cconeOrientation))

    def getConeOrientation(self, p: int) -> ArrayInt:
        """Return the orientations on the in-edges for this point in the DAG.

        Not collective.

        Parameters
        ----------
        p
            The point, which must lie in the chart set with `DMPlex.setChart`.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.getCone, DMPlex.setCone
        DMPlex.setChart, petsc.DMPlexGetConeOrientation

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR(DMPlexGetChart(self.dm, &pStart, &pEnd))
        assert cp>=pStart and cp<pEnd
        cdef PetscInt        norie = 0
        cdef const PetscInt *iorie = NULL
        CHKERR(DMPlexGetConeSize(self.dm, cp, &norie))
        CHKERR(DMPlexGetConeOrientation(self.dm, cp, &iorie))
        return array_i(norie, iorie)

    def setConeOrientation(self, p: int, orientation: Sequence[int]) -> None:
        """Set the orientations on the in-edges for this point in the DAG.

        Not collective.

        Parameters
        ----------
        p
            The point, which must lie in the chart set with `DMPlex.setChart`.
        orientation
            An array of orientations.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.getConeOrientation, DMPlex.setCone
        DMPlex.setChart, DMPlex.setConeSize, DM.setUp
        petsc.DMPlexSetConeOrientation

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR(DMPlexGetChart(self.dm, &pStart, &pEnd))
        assert cp>=pStart and cp<pEnd
        cdef PetscInt ncone = 0
        CHKERR(DMPlexGetConeSize(self.dm, cp, &ncone))
        cdef PetscInt  norie = 0
        cdef PetscInt *iorie = NULL
        orientation = iarray_i(orientation, &norie, &iorie)
        assert norie == ncone
        CHKERR(DMPlexSetConeOrientation(self.dm, cp, iorie))

    def setCellType(self, p: int, ctype: DM.PolytopeType) -> None:
        """Set the polytope type of a given cell.

        Not collective.

        Parameters
        ----------
        p
            The cell.
        ctype
            The polytope type of the cell.

        See Also
        --------
        DM, DMPlex, DMPlex.getCellTypeLabel, DMPlex.getDepth, DM.createLabel
        petsc.DMPlexSetCellType

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscDMPolytopeType val = ctype
        CHKERR(DMPlexSetCellType(self.dm, cp, val))

    def getCellType(self, p: int) -> DM.PolytopeType:
        """Return the polytope type of a given cell.

        Not collective.

        Parameters
        ----------
        p
            The cell.

        See Also
        --------
        DM, DMPlex, DM.PolytopeType, DMPlex.getCellTypeLabel, DMPlex.getDepth
        petsc.DMPlexGetCellType

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscDMPolytopeType ctype = DM_POLYTOPE_UNKNOWN
        CHKERR(DMPlexGetCellType(self.dm, cp, &ctype))
        return toInt(ctype)

    def getCellTypeLabel(self) -> DMLabel:
        """Return the `DMLabel` recording the polytope type of each cell.

        Not collective.

        See Also
        --------
        DM, DMPlex, DMPlex.getCellType, DM.createLabel
        petsc.DMPlexGetCellTypeLabel

        """
        cdef DMLabel label = DMLabel()
        CHKERR(DMPlexGetCellTypeLabel(self.dm, &label.dmlabel))
        CHKERR(PetscINCREF(label.obj))
        return label

    def getSupportSize(self, p: int) -> int:
        """Return the number of out-edges for this point in the DAG.

        Not collective.

        Parameters
        ----------
        p
            The point, which must lie in the chart set with `DMPlex.setChart`.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.setConeSize, DMPlex.setChart
        DMPlex.getConeSize, petsc.DMPlexGetSupportSize

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR(DMPlexGetChart(self.dm, &pStart, &pEnd))
        assert cp>=pStart and cp<pEnd
        cdef PetscInt ssize = 0
        CHKERR(DMPlexGetSupportSize(self.dm, cp, &ssize))
        return toInt(ssize)

    def setSupportSize(self, p: int, size: int) -> None:
        """Set the number of out-edges for this point in the DAG.

        Not collective.

        Parameters
        ----------
        p
            The point, which must lie in the chart set with `DMPlex.setChart`.
        size
            The support size for point ``p``.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.getSupportSize, DMPlex.setChart
        petsc.DMPlexSetSupportSize

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR(DMPlexGetChart(self.dm, &pStart, &pEnd))
        assert cp>=pStart and cp<pEnd
        cdef PetscInt ssize = asInt(size)
        CHKERR(DMPlexSetSupportSize(self.dm, cp, ssize))

    def getSupport(self, p: int) -> ArrayInt:
        """Return the points on the out-edges for this point in the DAG.

        Not collective.

        Parameters
        ----------
        p
            The point, which must lie in the chart set with `DMPlex.setChart`.

        See Also
        --------
        DM, DMPlex, DMPlex.getSupportSize, DMPlex.setSupport, DMPlex.getCone
        DMPlex.setChart, petsc.DMPlexGetSupport

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR(DMPlexGetChart(self.dm, &pStart, &pEnd))
        assert cp>=pStart and cp<pEnd
        cdef PetscInt        nsupp = 0
        cdef const PetscInt *isupp = NULL
        CHKERR(DMPlexGetSupportSize(self.dm, cp, &nsupp))
        CHKERR(DMPlexGetSupport(self.dm, cp, &isupp))
        return array_i(nsupp, isupp)

    def setSupport(self, p: int, supp: Sequence[int]) -> None:
        """Set the points on the out-edges for this point in the DAG.

        Not collective.

        Parameters
        ----------
        p
            The point, which must lie in the chart set with `DMPlex.setChart`.
        supp
            An array of points which are on the out-edges for point ``p``.

        See Also
        --------
        DM, DMPlex, DMPlex.setCone, DMPlex.setConeSize, DMPlex.create
        DMPlex.getSupport, DMPlex.setChart, DMPlex.setSupportSize, DM.setUp
        petsc.DMPlexSetSupport

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR(DMPlexGetChart(self.dm, &pStart, &pEnd))
        assert cp>=pStart and cp<pEnd
        cdef PetscInt  nsupp = 0
        cdef PetscInt *isupp = NULL
        supp = iarray_i(supp, &nsupp, &isupp)
        CHKERR(DMPlexSetSupportSize(self.dm, cp, nsupp))
        CHKERR(DMPlexSetSupport(self.dm, cp, isupp))

    def getMaxSizes(self) -> tuple[int, int]:
        """Return the maximum number of in-edges and out-edges of the DAG.

        Not collective.

        Returns
        -------
        maxConeSize : int
            The maximum number of in-edges.
        maxSupportSize : int
            The maximum number of out-edges.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.setConeSize, DMPlex.setChart
        petsc.DMPlexGetMaxSizes

        """
        cdef PetscInt maxConeSize = 0, maxSupportSize = 0
        CHKERR(DMPlexGetMaxSizes(self.dm, &maxConeSize, &maxSupportSize))
        return toInt(maxConeSize), toInt(maxSupportSize)

    def symmetrize(self) -> None:
        """Create support (out-edge) information from cone (in-edge) information.

        Not collective.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.setChart, DMPlex.setConeSize
        DMPlex.setCone, petsc.DMPlexSymmetrize

        """
        CHKERR(DMPlexSymmetrize(self.dm))

    def stratify(self) -> None:
        """Calculate the strata of DAG.

        Collective.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.symmetrize, petsc.DMPlexStratify

        """
        CHKERR(DMPlexStratify(self.dm))

    def orient(self) -> None:
        """Give a consistent orientation to the input mesh.

        Collective.

        See Also
        --------
        DM, DMPlex, DM.create, petsc.DMPlexOrient

        """
        CHKERR(DMPlexOrient(self.dm))

    def getCellNumbering(self) -> IS:
        """Return a global cell numbering for all cells on this process.

        Collective the first time it is called.

        See Also
        --------
        DM, DMPlex, DMPlex.getVertexNumbering, petsc.DMPlexGetCellNumbering

        """
        cdef IS iset = IS()
        CHKERR(DMPlexGetCellNumbering(self.dm, &iset.iset))
        CHKERR(PetscINCREF(iset.obj))
        return iset

    def getVertexNumbering(self) -> IS:
        """Return a global vertex numbering for all vertices on this process.

        Collective the first time it is called.

        See Also
        --------
        DM, DMPlex, DMPlex.getCellNumbering, petsc.DMPlexGetVertexNumbering

        """
        cdef IS iset = IS()
        CHKERR(DMPlexGetVertexNumbering(self.dm, &iset.iset))
        CHKERR(PetscINCREF(iset.obj))
        return iset

    def createPointNumbering(self) -> IS:
        """Create a global numbering for all points.

        Collective.

        See Also
        --------
        DM, DMPlex, DMPlex.getCellNumbering, petsc.DMPlexCreatePointNumbering

        """
        cdef IS iset = IS()
        CHKERR(DMPlexCreatePointNumbering(self.dm, &iset.iset))
        return iset

    def getDepth(self) -> int:
        """Return the depth of the DAG representing this mesh.

        Not collective.

        See Also
        --------
        DM, DMPlex, DMPlex.getDepthStratum, DMPlex.symmetrize
        petsc.DMPlexGetDepth

        """
        cdef PetscInt depth = 0
        CHKERR(DMPlexGetDepth(self.dm, &depth))
        return toInt(depth)

    def getDepthStratum(self, svalue: int) -> tuple[int, int]:
        """Return the bounds [``start``, ``end``) for all points at a certain depth.

        Not collective.

        Parameters
        ----------
        svalue
            The requested depth.

        Returns
        -------
        pStart : int
            The first stratum point.
        pEnd : int
            The upper bound for stratum points.

        See Also
        --------
        DM, DMPlex, DMPlex.getHeightStratum, DMPlex.getDepth
        DMPlex.symmetrize, DMPlex.interpolate, petsc.DMPlexGetDepthStratum

        """
        cdef PetscInt csvalue = asInt(svalue), sStart = 0, sEnd = 0
        CHKERR(DMPlexGetDepthStratum(self.dm, csvalue, &sStart, &sEnd))
        return (toInt(sStart), toInt(sEnd))

    def getHeightStratum(self, svalue: int) -> tuple[int, int]:
        """Return the bounds [``start``, ``end``) for all points at a certain height.

        Not collective.

        Parameters
        ----------
        svalue
            The requested height.

        Returns
        -------
        pStart : int
            The first stratum point.
        pEnd : int
            The upper bound for stratum points.

        See Also
        --------
        DM, DMPlex, DMPlex.getDepthStratum, DMPlex.getDepth
        petsc.DMPlexGetHeightStratum

        """
        cdef PetscInt csvalue = asInt(svalue), sStart = 0, sEnd = 0
        CHKERR(DMPlexGetHeightStratum(self.dm, csvalue, &sStart, &sEnd))
        return (toInt(sStart), toInt(sEnd))

    def getPointDepth(self, point: int) -> int:
        """Return the *depth* of a given point.

        Not collective.

        Parameters
        ----------
        point
            The point.

        See Also
        --------
        DM, DMPlex, DMPlex.getDepthStratum, DMPlex.getDepth
        petsc.DMPlexGetPointDepth

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt depth = 0
        CHKERR(DMPlexGetPointDepth(self.dm, cpoint, &depth))
        return toInt(depth)

    def getPointHeight(self, point: int) -> int:
        """Return the *height* of a given point.

        Not collective.

        Parameters
        ----------
        point
            The point.

        See Also
        --------
        DM, DMPlex, DMPlex.getHeightStratum
        petsc.DMPlexGetPointHeight

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt height = 0
        CHKERR(DMPlexGetPointHeight(self.dm, cpoint, &height))
        return toInt(height)

    def getMeet(self, points: Sequence[int]) -> ArrayInt:
        """Return an array for the meet of the set of points.

        Not collective.

        Parameters
        ----------
        points
            The input points.

        See Also
        --------
        DM, DMPlex, DMPlex.getJoin, petsc.DMPlexGetMeet

        """
        cdef PetscInt  numPoints = 0
        cdef PetscInt *ipoints = NULL
        cdef PetscInt  numCoveringPoints = 0
        cdef const PetscInt *coveringPoints = NULL
        points = iarray_i(points, &numPoints, &ipoints)
        CHKERR(DMPlexGetMeet(self.dm, numPoints, ipoints, &numCoveringPoints, &coveringPoints))
        try:
            return array_i(numCoveringPoints, coveringPoints)
        finally:
            CHKERR(DMPlexRestoreMeet(self.dm, numPoints, ipoints, &numCoveringPoints, &coveringPoints))

    def getJoin(self, points: Sequence[int]) -> ArrayInt:
        """Return an array for the join of the set of points.

        Not collective.

        Parameters
        ----------
        points
            The input points.

        See Also
        --------
        DM, DMPlex, DMPlex.getMeet, petsc.DMPlexGetJoin

        """
        cdef PetscInt  numPoints = 0
        cdef PetscInt *ipoints = NULL
        cdef PetscInt  numCoveringPoints = 0
        cdef const PetscInt *coveringPoints = NULL
        points = iarray_i(points, &numPoints, &ipoints)
        CHKERR(DMPlexGetJoin(self.dm, numPoints, ipoints, &numCoveringPoints, &coveringPoints))
        try:
            return array_i(numCoveringPoints, coveringPoints)
        finally:
            CHKERR(DMPlexRestoreJoin(self.dm, numPoints, ipoints, &numCoveringPoints, &coveringPoints))

    def getFullJoin(self, points: Sequence[int]) -> ArrayInt:
        """Return an array for the join of the set of points.

        Not collective.

        Parameters
        ----------
        points
            The input points.

        See Also
        --------
        DM, DMPlex, DMPlex.getJoin, DMPlex.getMeet, petsc.DMPlexGetFullJoin

        """
        cdef PetscInt  numPoints = 0
        cdef PetscInt *ipoints = NULL
        cdef PetscInt  numCoveringPoints = 0
        cdef const PetscInt *coveringPoints = NULL
        points = iarray_i(points, &numPoints, &ipoints)
        CHKERR(DMPlexGetFullJoin(self.dm, numPoints, ipoints, &numCoveringPoints, &coveringPoints))
        try:
            return array_i(numCoveringPoints, coveringPoints)
        finally:
            CHKERR(DMPlexRestoreJoin(self.dm, numPoints, ipoints, &numCoveringPoints, &coveringPoints))

    def getTransitiveClosure(self, p: int, useCone: bool | None = True) -> tuple[ArrayInt, ArrayInt]:
        """Return the points and orientations on the transitive closure of this point.

        Not collective.

        Parameters
        ----------
        p
            The mesh point.
        useCone
            `True` for the closure, otherwise return the star.

        Returns
        -------
        points : ArrayInt
            The points.
        orientations : ArrayInt
            The orientations.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DMPlex.setCone, DMPlex.setChart
        DMPlex.getCone, petsc.DMPlexGetTransitiveClosure

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR(DMPlexGetChart(self.dm, &pStart, &pEnd))
        assert cp>=pStart and cp<pEnd
        cdef PetscBool cuseCone = useCone
        cdef PetscInt  numPoints = 0
        cdef PetscInt *points = NULL
        CHKERR(DMPlexGetTransitiveClosure(self.dm, cp, cuseCone, &numPoints, &points))
        try:
            out = array_i(2*numPoints, points)
        finally:
            CHKERR(DMPlexRestoreTransitiveClosure(self.dm, cp, cuseCone, &numPoints, &points))
        return out[::2], out[1::2]

    def vecGetClosure(self, Section sec, Vec vec, p: int) -> ArrayScalar:
        """Return an array of values on the closure of ``p``.

        Not collective.

        Parameters
        ----------
        sec
            The section describing the layout in ``vec``.
        vec
            The local vector.
        p
            The point in the `DMPlex`.

        See Also
        --------
        DM, DMPlex, petsc.DMPlexVecRestoreClosure

        """
        cdef PetscInt cp = asInt(p), csize = 0
        cdef PetscScalar *cvals = NULL
        CHKERR(DMPlexVecGetClosure(self.dm, sec.sec, vec.vec, cp, &csize, &cvals))
        try:
            closure = array_s(csize, cvals)
        finally:
            CHKERR(DMPlexVecRestoreClosure(self.dm, sec.sec, vec.vec, cp, &csize, &cvals))
        return closure

    def getVecClosure(self, Section sec or None, Vec vec, point: int) -> ArrayScalar:
        """Return an array of the values on the closure of a point.

        Not collective.

        Parameters
        ----------
        sec
            The `Section` describing the layout in ``vec``
            or `None` to use the default section.
        vec
            The local vector.
        point
            The point in the `DMPlex`.

        See Also
        --------
        DM, DMPlex, petsc.DMPlexVecRestoreClosure

        """
        cdef PetscSection csec = sec.sec if sec is not None else NULL
        cdef PetscInt cp = asInt(point), csize = 0
        cdef PetscScalar *cvals = NULL
        CHKERR(DMPlexVecGetClosure(self.dm, csec, vec.vec, cp, &csize, &cvals))
        try:
            closure = array_s(csize, cvals)
        finally:
            CHKERR(DMPlexVecRestoreClosure(self.dm, csec, vec.vec, cp, &csize, &cvals))
        return closure

    def setVecClosure(self, Section sec or None, Vec vec, point: int, values: Sequence[Scalar], addv: InsertModeSpec | None = None) -> None:
        """Set an array of the values on the closure of ``point``.

        Not collective.

        Parameters
        ----------
        sec
            The section describing the layout in ``vec``,
            or `None` to use the default section.
        vec
            The local vector.
        point
            The point in the `DMPlex`.
        values
            The array of values.
        mode
            The insertion mode.

        See Also
        --------
        DM, DMPlex, petsc.DMPlexVecSetClosure

        """
        cdef PetscSection csec = sec.sec if sec is not None else NULL
        cdef PetscInt cp = asInt(point)
        cdef PetscInt csize = 0
        cdef PetscScalar *cvals = NULL
        cdef object unused = iarray_s(values, &csize, &cvals)
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR(DMPlexVecSetClosure(self.dm, csec, vec.vec, cp, cvals, im))

    def setMatClosure(self, Section sec or None, Section gsec or None,
                      Mat mat, point: int, values: Sequence[Scalar], addv: InsertModeSpec | None = None) -> None:
        """Set an array of the values on the closure of ``point``.

        Not collective.

        Parameters
        ----------
        sec
            The section describing the layout in ``mat``,
            or `None` to use the default section.
        gsec
            The section describing the layout in ``mat``,
            or `None` to use the default global section.
        mat
            The matrix.
        point
            The point in the `DMPlex`.
        values
            The array of values.
        mode
            The insertion mode.

        See Also
        --------
        DM, DMPlex, petsc.DMPlexMatSetClosure

        """
        cdef PetscSection csec = sec.sec if sec is not None else NULL
        cdef PetscSection cgsec = gsec.sec if gsec is not None else NULL
        cdef PetscInt cp = asInt(point)
        cdef PetscInt csize = 0
        cdef PetscScalar *cvals = NULL
        cdef object unused = iarray_s(values, &csize, &cvals)
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR(DMPlexMatSetClosure(self.dm, csec, cgsec, mat.mat, cp, cvals, im))

    def generate(self, DMPlex boundary, name: str | None = None, interpolate: bool | None = True) -> Self:
        """Generate a mesh.

        Not collective.

        Parameters
        ----------
        boundary
            The `DMPlex` boundary object.
        name
            The mesh generation package name.
        interpolate
            Flag to create intermediate mesh elements.

        See Also
        --------
        DM, DMPlex, DMPlex.create, DM.refine, petsc_options
        petsc.DMPlexGenerate

        """
        cdef PetscBool interp = interpolate
        cdef const char *cname = NULL
        if name: name = str2bytes(name, &cname)
        cdef PetscDM   newdm = NULL
        CHKERR(DMPlexGenerate(boundary.dm, cname, interp, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm
        return self

    def setTriangleOptions(self, opts: str) -> None:
        """Set the options used for the Triangle mesh generator.

        Not collective.

        Parameters
        ----------
        opts
            The command line options.

        See Also
        --------
        petsc_options, DM, DMPlex, DMPlex.setTetGenOptions, DMPlex.generate
        petsc.DMPlexTriangleSetOptions

        """
        cdef const char *copts = NULL
        opts = str2bytes(opts, &copts)
        CHKERR(DMPlexTriangleSetOptions(self.dm, copts))

    def setTetGenOptions(self, opts: str) -> None:
        """Set the options used for the Tetgen mesh generator.

        Not collective.

        Parameters
        ----------
        opts
            The command line options.

        See Also
        --------
        petsc_options, DM, DMPlex, DMPlex.setTriangleOptions, DMPlex.generate
        petsc.DMPlexTetgenSetOptions

        """
        cdef const char *copts = NULL
        opts = str2bytes(opts, &copts)
        CHKERR(DMPlexTetgenSetOptions(self.dm, copts))

    def markBoundaryFaces(self, label: str, value: int | None = None) -> DMLabel:
        """Mark all faces on the boundary.

        Not collective.

        Parameters
        ----------
        value
            The marker value, or `DETERMINE` or `None` to use some
            value in the closure (or 1 if none are found).

        See Also
        --------
        DM, DMPlex, DMLabel.create, DM.createLabel
        petsc.DMPlexMarkBoundaryFaces

        """
        cdef PetscInt ival = PETSC_DETERMINE
        if value is not None: ival = asInt(value)
        if not self.hasLabel(label):
            self.createLabel(label)
        cdef const char *cval = NULL
        label = str2bytes(label, &cval)
        cdef PetscDMLabel clbl = NULL
        CHKERR(DMGetLabel(self.dm, cval, &clbl))
        CHKERR(DMPlexMarkBoundaryFaces(self.dm, ival, clbl))

    def labelComplete(self, DMLabel label) -> None:
        """Add the transitive closure to the surface.

        Not collective.

        Parameters
        ----------
        label
            A `DMLabel` marking the surface points.

        See Also
        --------
        DM, DMPlex, DMPlex.labelCohesiveComplete, petsc.DMPlexLabelComplete

        """
        CHKERR(DMPlexLabelComplete(self.dm, label.dmlabel))

    def labelCohesiveComplete(self, DMLabel label, DMLabel bdlabel, bdvalue: int,
                              flip: bool, split: bool, DMPlex subdm) -> None:
        """Add all other mesh pieces to complete the surface.

        Not collective.

        Parameters
        ----------
        label
            A `DMLabel` marking the surface.
        bdlabel
            A `DMLabel` marking the vertices on the boundary
            which will not be duplicated.
        bdvalue
            Value of `DMLabel` marking the vertices on the boundary.
        flip
            Flag to flip the submesh normal and replace points
            on the other side.
        split
            Flag to split faces incident on the surface boundary,
            rather than clamping those faces to the boundary
        subdm
            The `DMPlex` associated with the label.

        See Also
        --------
        DM, DMPlex, DMPlex.labelComplete
        petsc.DMPlexLabelCohesiveComplete

        """
        cdef PetscBool flg  = flip
        cdef PetscBool flg2 = split
        cdef PetscInt  val  = asInt(bdvalue)
        CHKERR(DMPlexLabelCohesiveComplete(self.dm, label.dmlabel, bdlabel.dmlabel, val, flg, flg2, subdm.dm))

    def setAdjacencyUseAnchors(self, useAnchors: bool = True) -> None:
        """Define adjacency in the mesh using the point-to-point constraints.

        Logically collective.

        Parameters
        ----------
        useAnchors
            Flag to use the constraints.
            If `True`, then constrained points are omitted from `DMPlex.getAdjacency`,
            and their anchor points appear in their place.

        See Also
        --------
        DMPlex, DMPlex.getAdjacency, DMPlex.distribute
        petsc.DMPlexSetAdjacencyUseAnchors

        """
        cdef PetscBool flag = useAnchors
        CHKERR(DMPlexSetAdjacencyUseAnchors(self.dm, flag))

    def getAdjacencyUseAnchors(self) -> bool:
        """Query whether adjacency in the mesh uses the point-to-point constraints.

        Not collective.

        See Also
        --------
        DMPlex, DMPlex.getAdjacency, DMPlex.distribute
        petsc.DMPlexGetAdjacencyUseAnchors

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(DMPlexGetAdjacencyUseAnchors(self.dm, &flag))
        return toBool(flag)

    def getAdjacency(self, p: int) -> ArrayInt:
        """Return all points adjacent to the given point.

        Not collective.

        Parameters
        ----------
        p
            The point.

        See Also
        --------
        DMPlex, DMPlex.distribute, petsc.DMPlexGetAdjacency

        """
        cdef PetscInt cp = asInt(p)
        cdef PetscInt nadj = PETSC_DETERMINE
        cdef PetscInt *iadj = NULL
        CHKERR(DMPlexGetAdjacency(self.dm, cp, &nadj, &iadj))
        try:
            adjacency = array_i(nadj, iadj)
        finally:
            CHKERR(PetscFree(iadj))
        return adjacency

    def setPartitioner(self, Partitioner part) -> None:
        """Set the mesh partitioner.

        Logically collective.

        Parameters
        ----------
        part
            The partitioner.

        See Also
        --------
        DM, DMPlex, Partitioner, DMPlex.distribute, DMPlex.getPartitioner
        Partitioner.create, petsc.DMPlexSetPartitioner

        """
        CHKERR(DMPlexSetPartitioner(self.dm, part.part))

    def getPartitioner(self) -> Partitioner:
        """Return the mesh partitioner.

        Not collective.

        See Also
        --------
        DM, DMPlex, Partitioner, Section, DMPlex.distribute
        DMPlex.setPartitioner, Partitioner.create
        petsc.PetscPartitionerDMPlexPartition, petsc.DMPlexGetPartitioner

        """
        cdef Partitioner part = Partitioner()
        CHKERR(DMPlexGetPartitioner(self.dm, &part.part))
        CHKERR(PetscINCREF(part.obj))
        return part

    def rebalanceSharedPoints(self, entityDepth: int | None = 0, useInitialGuess: bool | None = True, parallel: bool | None = True) -> bool:
        """Redistribute shared points in order to achieve better balancing.

        Collective.

        Parameters
        ----------
        entityDepth
            Depth of the entity to balance (e.g., 0 -> balance vertices).
        useInitialGuess
            Whether to use the current distribution as initial guess.
        parallel
            Whether to use ParMETIS and do the partition in parallel
            or gather the graph onto a single process.

        Returns
        -------
        success : bool
            Whether the graph partitioning was successful or not.
            Unsuccessful simply means no change to the partitioning.

        See Also
        --------
        DM, DMPlex, DMPlex.distribute, petsc_options
        petsc.DMPlexRebalanceSharedPoints

        """
        cdef PetscInt centityDepth = asInt(entityDepth)
        cdef PetscBool cuseInitialGuess = asBool(useInitialGuess)
        cdef PetscBool cparallel = asBool(parallel)
        cdef PetscBool csuccess = PETSC_FALSE
        CHKERR(DMPlexRebalanceSharedPoints(self.dm, centityDepth, cuseInitialGuess, cparallel, &csuccess))
        return toBool(csuccess)

    def distribute(self, overlap: int | None = 0) -> SF | None:
        """Distribute the mesh and any associated sections.

        Collective.

        Parameters
        ----------
        overlap
            The overlap of partitions.

        Returns
        -------
        sf : SF or None
            The `SF` used for point distribution, or `None` if not distributed.

        See Also
        --------
        DM, DMPlex, DMPlex.create, petsc.DMPlexDistribute

        """
        cdef PetscDM dmParallel = NULL
        cdef PetscInt coverlap = asInt(overlap)
        cdef SF sf = SF()
        CHKERR(DMPlexDistribute(self.dm, coverlap, &sf.sf, &dmParallel))
        if dmParallel != NULL:
            CHKERR(PetscCLEAR(self.obj)); self.dm = dmParallel
            return sf

    def distributeOverlap(self, overlap: int | None = 0) -> SF:
        """Add partition overlap to a distributed non-overlapping `DMPlex`.

        Collective.

        Parameters
        ----------
        overlap
            The overlap of partitions (the same on all ranks).

        Returns
        -------
        sf : SF
            The `SF` used for point distribution.

        See Also
        --------
        DM, DMPlex, SF, DMPlex.create, DMPlex.distribute,
        petsc.DMPlexDistributeOverlap

        """
        cdef PetscInt coverlap = asInt(overlap)
        cdef SF sf = SF()
        cdef PetscDM dmOverlap = NULL
        CHKERR(DMPlexDistributeOverlap(self.dm, coverlap,
                                       &sf.sf, &dmOverlap))
        CHKERR(PetscCLEAR(self.obj)); self.dm = dmOverlap
        return sf

    def isDistributed(self) -> bool:
        """Return the flag indicating if the mesh is distributed.

        Collective.

        See Also
        --------
        DM, DMPlex, DMPlex.distribute, petsc.DMPlexIsDistributed

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(DMPlexIsDistributed(self.dm, &flag))
        return toBool(flag)

    def isSimplex(self) -> bool:
        """Return the flag indicating if the first cell is a simplex.

        Not collective.

        See Also
        --------
        DM, DMPlex, DMPlex.getCellType, DMPlex.getHeightStratum
        petsc.DMPlexIsSimplex

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(DMPlexIsSimplex(self.dm, &flag))
        return toBool(flag)

    def distributeGetDefault(self) -> bool:
        """Return a flag indicating whether the `DM` should be distributed by default.

        Not collective.

        Returns
        -------
        dist : bool
            Flag indicating whether the `DMPlex` should be distributed by default.

        See Also
        --------
        DM, DMPlex, DMPlex.distributeSetDefault, DMPlex.distribute
        petsc.DMPlexDistributeGetDefault

        """
        cdef PetscBool dist = PETSC_FALSE
        CHKERR(DMPlexDistributeGetDefault(self.dm, &dist))
        return toBool(dist)

    def distributeSetDefault(self, flag: bool) -> None:
        """Set flag indicating whether the `DMPlex` should be distributed by default.

        Logically collective.

        Parameters
        ----------
        flag
            Flag indicating whether the `DMPlex` should be distributed by default.

        See Also
        --------
        DMPlex, DMPlex.distributeGetDefault, DMPlex.distribute
        petsc.DMPlexDistributeSetDefault

        """
        cdef PetscBool dist = asBool(flag)
        CHKERR(DMPlexDistributeSetDefault(self.dm, dist))
        return

    def distributionSetName(self, name: str) -> None:
        """Set the name of the specific parallel distribution.

        Logically collective.

        Parameters
        ----------
        name
            The name of the specific parallel distribution.

        See Also
        --------
        DMPlex, DMPlex.distributionGetName, DMPlex.topologyView
        DMPlex.topologyLoad, petsc.DMPlexDistributionSetName

        """
        cdef const char *cname = NULL
        if name is not None:
            name = str2bytes(name, &cname)
        CHKERR(DMPlexDistributionSetName(self.dm, cname))

    def distributionGetName(self) -> str:
        """Retrieve the name of the specific parallel distribution.

        Not collective.

        Returns
        -------
        name : str
            The name of the specific parallel distribution.

        See Also
        --------
        DMPlex, DMPlex.distributionSetName, DMPlex.topologyView
        DMPlex.topologyLoad, petsc.DMPlexDistributionGetName

        """
        cdef const char *cname = NULL
        CHKERR(DMPlexDistributionGetName(self.dm, &cname))
        return bytes2str(cname)

    def interpolate(self) -> None:
        """Convert to a mesh with all intermediate faces, edges, etc.

        Collective.

        See Also
        --------
        DMPlex, DMPlex.uninterpolate, DMPlex.createFromCellList
        petsc.DMPlexInterpolate

        """
        cdef PetscDM newdm = NULL
        CHKERR(DMPlexInterpolate(self.dm, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm

    def uninterpolate(self) -> None:
        """Convert to a mesh with only cells and vertices.

        Collective.

        See Also
        --------
        DMPlex, DMPlex.interpolate, DMPlex.createFromCellList
        petsc.DMPlexUninterpolate

        """
        cdef PetscDM newdm = NULL
        CHKERR(DMPlexUninterpolate(self.dm, &newdm))
        CHKERR(PetscCLEAR(self.obj)); self.dm = newdm

    def distributeField(self, SF sf, Section sec, Vec vec,
                        Section newsec=None, Vec newvec=None) -> tuple[Section, Vec]:
        """Distribute field data with a with a given `SF`.

        Collective.

        Parameters
        ----------
        sf
            The `SF` describing the communication pattern.
        sec
            The `Section` for existing data layout.
        vec
            The existing data in a local vector.
        newsec
            The `SF` describing the new data layout.
        newvec
            The new data in a local vector.

        Returns
        -------
        newSection : Section
            The `SF` describing the new data layout.
        newVec : Vec
            The new data in a local vector.

        See Also
        --------
        DMPlex, DMPlex.distribute, petsc.DMPlexDistributeField

        """
        cdef MPI_Comm ccomm = MPI_COMM_NULL
        if newsec is None: newsec = Section()
        if newvec is None: newvec = Vec()
        if newsec.sec == NULL:
            CHKERR(PetscObjectGetComm(<PetscObject>sec.sec, &ccomm))
            CHKERR(PetscSectionCreate(ccomm, &newsec.sec))
        if newvec.vec == NULL:
            CHKERR(PetscObjectGetComm(<PetscObject>vec.vec, &ccomm))
            CHKERR(VecCreate(ccomm, &newvec.vec))
        CHKERR(DMPlexDistributeField(self.dm, sf.sf,
                                     sec.sec, vec.vec,
                                     newsec.sec, newvec.vec))
        return (newsec, newvec)

    def getMinRadius(self) -> float:
        """Return the minimum distance from any cell centroid to a face.

        Not collective.

        See Also
        --------
        DMPlex, DM.getCoordinates, petsc.DMPlexGetMinRadius

        """
        cdef PetscReal cminradius = 0.
        CHKERR(DMPlexGetMinRadius(self.dm, &cminradius))
        return asReal(cminradius)

    def createCoarsePointIS(self) -> IS:
        """Create an `IS` covering the coarse `DMPlex` chart with the fine points as data.

        Collective.

        Returns
        -------
        fpointIS : IS
            The `IS` of all the fine points which exist in the original coarse mesh.

        See Also
        --------
        DM, DMPlex, IS, DM.refine, DMPlex.setRefinementUniform
        petsc.DMPlexCreateCoarsePointIS

        """
        cdef IS fpoint = IS()
        CHKERR(DMPlexCreateCoarsePointIS(self.dm, &fpoint.iset))
        return fpoint

    def createSection(self, numComp: Sequence[int], numDof: Sequence[int],
                      bcField: Sequence[int] | None = None, bcComps: Sequence[IS] | None = None, bcPoints: Sequence[IS] | None = None,
                      IS perm=None) -> Section:
        """Create a `Section` based upon the DOF layout specification provided.

        Not collective.

        Parameters
        ----------
        numComp
            An array of size ``numFields`` holding the number of components per field.
        numDof
            An array of size ``numFields*(dim+1)`` holding the number of DOFs
            per field on a mesh piece of dimension ``dim``.
        bcField
            An array of size ``numBC`` giving the field number for each boundary
            condition, where ``numBC`` is the number of boundary conditions.
        bcComps
            An array of size ``numBC`` giving an `IS` holding the field
            components to which each boundary condition applies.
        bcPoints
            An array of size ``numBC`` giving an `IS` holding the `DMPlex` points
            to which each boundary condition applies.
        perm
            Permutation of the chart.

        See Also
        --------
        DM, DMPlex, DMPlex.create, Section.create, Section.setPermutation
        petsc.DMPlexCreateSection

        """
        # topological dimension
        cdef PetscInt dim = 0
        CHKERR(DMGetDimension(self.dm, &dim))
        # components and DOFs
        cdef PetscInt ncomp = 0, ndof = 0
        cdef PetscInt *icomp = NULL, *idof = NULL
        numComp = iarray_i(numComp, &ncomp, &icomp)
        numDof  = iarray_i(numDof, &ndof, &idof)
        assert ndof == ncomp*(dim+1)
        # boundary conditions
        cdef PetscInt nbc = 0, i = 0
        cdef PetscInt *bcfield = NULL
        cdef PetscIS *bccomps  = NULL
        cdef PetscIS *bcpoints = NULL
        cdef object unused1, unused2
        if bcField is not None:
            bcField = iarray_i(bcField, &nbc, &bcfield)
            if bcComps is not None:
                bcComps = list(bcComps)
                assert len(bcComps) == nbc
                unused1 = oarray_p(empty_p(nbc), NULL, <void**>&bccomps)
                for i from 0 <= i < nbc:
                    bccomps[i] = (<IS?>bcComps[<Py_ssize_t>i]).iset
            if bcPoints is not None:
                bcPoints = list(bcPoints)
                assert len(bcPoints) == nbc
                unused2 = oarray_p(empty_p(nbc), NULL, <void**>&bcpoints)
                for i from 0 <= i < nbc:
                    bcpoints[i] = (<IS?>bcPoints[<Py_ssize_t>i]).iset
            else:
                raise ValueError("bcPoints is a required argument")
        else:
            assert bcComps  is None
            assert bcPoints is None
        # optional chart permutations
        cdef PetscIS cperm = NULL
        if perm is not None: cperm = perm.iset
        # create section
        cdef Section sec = Section()
        CHKERR(DMPlexCreateSection(self.dm, NULL, icomp, idof,
                                   nbc, bcfield, bccomps, bcpoints,
                                   cperm, &sec.sec))
        return sec

    def getPointLocal(self, point: int) -> tuple[int, int]:
        """Return location of point data in local `Vec`.

        Not collective.

        Parameters
        ----------
        point
            The topological point.

        Returns
        -------
        start : int
            Start of point data.
        end : int
            End of point data.

        See Also
        --------
        DM, DMPlex, DMPlex.getPointLocalField, Section.getOffset
        Section.getDof, petsc.DMPlexGetPointLocal

        """
        cdef PetscInt start = 0, end = 0
        cdef PetscInt cpoint = asInt(point)
        CHKERR(DMPlexGetPointLocal(self.dm, cpoint, &start, &end))
        return toInt(start), toInt(end)

    def getPointLocalField(self, point: int, field: int) -> tuple[int, int]:
        """Return location of point field data in local `Vec`.

        Not collective.

        Parameters
        ----------
        point
            The topological point.
        field
            The field number.

        Returns
        -------
        start : int
            Start of point data.
        end : int
            End of point data.

        See Also
        --------
        DM, DMPlex, DMPlex.getPointLocal, Section.getOffset
        petsc.DMPlexGetPointLocalField

        """
        cdef PetscInt start = 0, end = 0
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        CHKERR(DMPlexGetPointLocalField(self.dm, cpoint, cfield, &start, &end))
        return toInt(start), toInt(end)

    def getPointGlobal(self, point: int) -> tuple[int, int]:
        """Return location of point data in global `Vec`.

        Not collective.

        Parameters
        ----------
        point
            The topological point.

        Returns
        -------
        start : int
            Start of point data; returns ``-(globalStart+1)`` if point is not owned.
        end : int
            End of point data; returns ``-(globalEnd+1)`` if point is not owned.

        See Also
        --------
        DM, DMPlex, DMPlex.getPointGlobalField, Section.getOffset
        Section.getDof, DMPlex.getPointLocal, petsc.DMPlexGetPointGlobal

        """
        cdef PetscInt start = 0, end = 0
        cdef PetscInt cpoint = asInt(point)
        CHKERR(DMPlexGetPointGlobal(self.dm, cpoint, &start, &end))
        return toInt(start), toInt(end)

    def getPointGlobalField(self, point: int, field: int) -> tuple[int, int]:
        """Return location of point field data in global `Vec`.

        Not collective.

        Parameters
        ----------
        point
            The topological point.
        field
            The field number.

        Returns
        -------
        start : int
            Start of point data; returns ``-(globalStart+1)`` if point is not owned.
        end : int
            End of point data; returns ``-(globalEnd+1)`` if point is not owned.

        See Also
        --------
        DM, DMPlex, DMPlex.getPointGlobal, Section.getOffset, Section.getDof
        DMPlex.getPointLocal, petsc.DMPlexGetPointGlobalField

        """
        cdef PetscInt start = 0, end = 0
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        CHKERR(DMPlexGetPointGlobalField(self.dm, cpoint, cfield, &start, &end))
        return toInt(start), toInt(end)

    def createClosureIndex(self, Section sec or None) -> None:
        """Calculate an index for ``sec`` for the closure operation.

        Not collective.

        Parameters
        ----------
        sec
            The `Section` describing the layout in the local vector,
            or `None` to use the default section.

        See Also
        --------
        DM, DMPlex, Section, DMPlex.vecGetClosure
        petsc.DMPlexCreateClosureIndex

        """
        cdef PetscSection csec = sec.sec if sec is not None else NULL
        CHKERR(DMPlexCreateClosureIndex(self.dm, csec))

    #

    def setRefinementUniform(self, refinementUniform: bool | None = True) -> None:
        """Set the flag for uniform refinement.

        Logically collective.

        Parameters
        ----------
        refinementUniform
            The flag for uniform refinement.

        See Also
        --------
        DM, DMPlex, DM.refine, DMPlex.getRefinementUniform
        DMPlex.getRefinementLimit, DMPlex.setRefinementLimit
        petsc.DMPlexSetRefinementUniform

        """
        cdef PetscBool flag = refinementUniform
        CHKERR(DMPlexSetRefinementUniform(self.dm, flag))

    def getRefinementUniform(self) -> bool:
        """Retrieve the flag for uniform refinement.

        Not collective.

        Returns
        -------
        refinementUniform : bool
            The flag for uniform refinement.

        See Also
        --------
        DM, DMPlex, DM.refine, DMPlex.setRefinementUniform
        DMPlex.getRefinementLimit, DMPlex.setRefinementLimit
        petsc.DMPlexGetRefinementUniform

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(DMPlexGetRefinementUniform(self.dm, &flag))
        return toBool(flag)

    def setRefinementLimit(self, refinementLimit: float) -> None:
        """Set the maximum cell volume for refinement.

        Logically collective.

        Parameters
        ----------
        refinementLimit
            The maximum cell volume in the refined mesh.

        See Also
        --------
        DM, DMPlex, DM.refine, DMPlex.getRefinementLimit
        DMPlex.getRefinementUniform, DMPlex.setRefinementUniform
        petsc.DMPlexSetRefinementLimit

        """
        cdef PetscReal rval = asReal(refinementLimit)
        CHKERR(DMPlexSetRefinementLimit(self.dm, rval))

    def getRefinementLimit(self) -> float:
        """Retrieve the maximum cell volume for refinement.

        Not collective.

        See Also
        --------
        DM, DMPlex, DM.refine, DMPlex.setRefinementLimit
        DMPlex.getRefinementUniform, DMPlex.setRefinementUniform
        petsc.DMPlexGetRefinementLimit

        """
        cdef PetscReal rval = 0.0
        CHKERR(DMPlexGetRefinementLimit(self.dm, &rval))
        return toReal(rval)

    def getOrdering(self, otype: Mat.OrderingType) -> IS:
        """Calculate a reordering of the mesh.

        Collective.

        Parameters
        ----------
        otype
            Type of reordering, see `Mat.OrderingType`.

        Returns
        -------
        perm : IS
            The point permutation.

        See Also
        --------
        DMPlex, DMPlex.permute, Mat.OrderingType, Mat.getOrdering
        petsc.DMPlexGetOrdering

        """
        cdef PetscMatOrderingType cval = NULL
        cdef PetscDMLabel label = NULL
        otype = str2bytes(otype, &cval)
        cdef IS perm = IS()
        CHKERR(DMPlexGetOrdering(self.dm, cval, label, &perm.iset))
        return perm

    def permute(self, IS perm) -> DMPlex:
        """Reorder the mesh according to the input permutation.

        Collective.

        Parameters
        ----------
        perm
            The point permutation, ``perm[old point number] = new point number``.

        Returns
        -------
        pdm : DMPlex
            The permuted `DMPlex`.

        See Also
        --------
        DMPlex, Mat.permute, petsc.DMPlexPermute

        """
        cdef DMPlex dm = <DMPlex>type(self)()
        CHKERR(DMPlexPermute(self.dm, perm.iset, &dm.dm))
        return dm

    def reorderGetDefault(self) -> DM.ReorderDefaultFlag:
        """Return flag indicating whether the `DMPlex` should be reordered by default.

        Not collective.

        See Also
        --------
        `DMPlex.reorderSetDefault`, petsc.DMPlexReorderGetDefault

        """
        cdef PetscDMReorderDefaultFlag reorder = DM_REORDER_DEFAULT_NOTSET
        CHKERR(DMPlexReorderGetDefault(self.dm, &reorder))
        return reorder

    def reorderSetDefault(self, flag: DM.ReorderDefaultFlag) -> None:
        """Set flag indicating whether the DM should be reordered by default.

        Logically collective.

        Parameters
        ----------
        reorder
            Flag for reordering.

        See Also
        --------
        DMPlex.reorderGetDefault, petsc.DMPlexReorderSetDefault

        """
        cdef PetscDMReorderDefaultFlag reorder = flag
        CHKERR(DMPlexReorderSetDefault(self.dm, reorder))
        return

    #

    def computeCellGeometryFVM(self, cell: int) -> tuple[float, ArrayReal, ArrayReal]:
        """Compute the volume for a given cell.

        Not collective.

        Parameters
        ----------
        cell
            The cell.

        Returns
        -------
        volume : float
            The cell volume.
        centroid : ArrayReal
            The cell centroid.
        normal : ArrayReal
            The cell normal, if appropriate.

        See Also
        --------
        DMPlex, DM.getCoordinateSection, DM.getCoordinates
        petsc.DMPlexComputeCellGeometryFVM

        """
        cdef PetscInt cdim = 0
        cdef PetscInt ccell = asInt(cell)
        CHKERR(DMGetCoordinateDim(self.dm, &cdim))
        cdef PetscReal vol = 0, centroid[3], normal[3]
        CHKERR(DMPlexComputeCellGeometryFVM(self.dm, ccell, &vol, centroid, normal))
        return (toReal(vol), array_r(cdim, centroid), array_r(cdim, normal))

    def constructGhostCells(self, labelName: str | None = None) -> int:
        """Construct ghost cells which connect to every boundary face.

        Collective.

        Parameters
        ----------
        labelName
            The name of the label specifying the boundary faces.
            Defaults to ``"Face Sets"``.

        Returns
        -------
        numGhostCells : int
            The number of ghost cells added to the `DMPlex`.

        See Also
        --------
        DM, DMPlex, DM.create, petsc.DMPlexConstructGhostCells

        """
        cdef const char *cname = NULL
        labelName = str2bytes(labelName, &cname)
        cdef PetscInt numGhostCells = 0
        cdef PetscDM dmGhosted = NULL
        CHKERR(DMPlexConstructGhostCells(self.dm, cname, &numGhostCells, &dmGhosted))
        CHKERR(PetscCLEAR(self.obj)); self.dm = dmGhosted
        return toInt(numGhostCells)

    def getSubpointIS(self) -> IS:
        """Return an `IS` covering the entire subdm chart.

        Not collective.

        Returns
        -------
        iset : IS
            The `IS` containing subdm's parent's points.

        See Also
        --------
        DM, DMPlex, petsc.DMPlexGetSubpointIS

        """
        cdef IS iset = IS()
        CHKERR(DMPlexGetSubpointIS(self.dm, &iset.iset))
        PetscINCREF(iset.obj)
        return iset

    def getSubpointMap(self) -> DMLabel:
        """Return a `DMLabel` with point dimension as values.

        Not collective.

        Returns
        -------
        label : DMLabel
            The `DMLabel` whose values are subdm's point dimensions.

        See Also
        --------
        DM, DMPlex, petsc.DMPlexGetSubpointMap

        """
        cdef DMLabel label = DMLabel()
        CHKERR(DMPlexGetSubpointMap(self.dm, &label.dmlabel))
        PetscINCREF(label.obj)
        return label

    # Metric

    def metricSetFromOptions(self) -> None:
        """Configure the object from the options database.

        Collective.

        See Also
        --------
        petsc_options

        """
        # FIXME petsc.DMPlexMetricSetFromOptions
        CHKERR(DMPlexMetricSetFromOptions(self.dm))

    def metricSetUniform(self, uniform: bool) -> None:
        """Record whether the metric is uniform or not.

        Logically collective.

        Parameters
        ----------
        uniform
            Flag indicating whether the metric is uniform or not.

        See Also
        --------
        DMPlex.metricIsUniform, DMPlex.metricSetIsotropic
        DMPlex.metricSetRestrictAnisotropyFirst, petsc.DMPlexMetricSetUniform

        """
        cdef PetscBool bval = asBool(uniform)
        CHKERR(DMPlexMetricSetUniform(self.dm, bval))

    def metricIsUniform(self) -> bool:
        """Return the flag indicating whether the metric is uniform or not.

        Not collective.

        See Also
        --------
        DMPlex.metricSetUniform, DMPlex.metricRestrictAnisotropyFirst
        petsc.DMPlexMetricIsUniform

        """
        cdef PetscBool uniform = PETSC_FALSE
        CHKERR(DMPlexMetricIsUniform(self.dm, &uniform))
        return toBool(uniform)

    def metricSetIsotropic(self, isotropic: bool) -> None:
        """Record whether the metric is isotropic or not.

        Logically collective.

        Parameters
        ----------
        isotropic
            Flag indicating whether the metric is isotropic or not.

        See Also
        --------
        DMPlex.metricIsIsotropic, DMPlex.metricSetUniform
        DMPlex.metricSetRestrictAnisotropyFirst, petsc.DMPlexMetricSetIsotropic

        """
        cdef PetscBool bval = asBool(isotropic)
        CHKERR(DMPlexMetricSetIsotropic(self.dm, bval))

    def metricIsIsotropic(self) -> bool:
        """Return the flag indicating whether the metric is isotropic or not.

        Not collective.

        See Also
        --------
        DMPlex.metricSetIsotropic, DMPlex.metricIsUniform
        DMPlex.metricRestrictAnisotropyFirst, petsc.DMPlexMetricIsIsotropic

        """
        cdef PetscBool isotropic = PETSC_FALSE
        CHKERR(DMPlexMetricIsIsotropic(self.dm, &isotropic))
        return toBool(isotropic)

    def metricSetRestrictAnisotropyFirst(self, restrictAnisotropyFirst: bool) -> None:
        """Record whether anisotropy is be restricted before normalization or after.

        Logically collective.

        Parameters
        ----------
        restrictAnisotropyFirst
            Flag indicating if anisotropy is restricted before normalization or after.

        See Also
        --------
        DMPlex.metricSetIsotropic, DMPlex.metricRestrictAnisotropyFirst
        petsc.DMPlexMetricSetRestrictAnisotropyFirst

        """
        cdef PetscBool bval = asBool(restrictAnisotropyFirst)
        CHKERR(DMPlexMetricSetRestrictAnisotropyFirst(self.dm, bval))

    def metricRestrictAnisotropyFirst(self) -> bool:
        """Return ``true`` if anisotropy is restricted before normalization.

        Not collective.

        See Also
        --------
        DMPlex.metricIsIsotropic, DMPlex.metricSetRestrictAnisotropyFirst
        petsc.DMPlexMetricRestrictAnisotropyFirst

        """
        cdef PetscBool restrictAnisotropyFirst = PETSC_FALSE
        CHKERR(DMPlexMetricRestrictAnisotropyFirst(self.dm, &restrictAnisotropyFirst))
        return toBool(restrictAnisotropyFirst)

    def metricSetNoInsertion(self, noInsert: bool) -> None:
        """Set the flag indicating whether node insertion should be turned off.

        Logically collective.

        Parameters
        ----------
        noInsert
            Flag indicating whether node insertion and deletion should be turned off.

        See Also
        --------
        DMPlex.metricNoInsertion, DMPlex.metricSetNoSwapping
        DMPlex.metricSetNoMovement, DMPlex.metricSetNoSurf
        petsc.DMPlexMetricSetNoInsertion

        """
        cdef PetscBool bval = asBool(noInsert)
        CHKERR(DMPlexMetricSetNoInsertion(self.dm, bval))

    def metricNoInsertion(self) -> bool:
        """Return the flag indicating whether node insertion and deletion are turned off.

        Not collective.

        See Also
        --------
        DMPlex.metricSetNoInsertion, DMPlex.metricNoSwapping
        DMPlex.metricNoMovement, DMPlex.metricNoSurf
        petsc.DMPlexMetricNoInsertion

        """
        cdef PetscBool noInsert = PETSC_FALSE
        CHKERR(DMPlexMetricNoInsertion(self.dm, &noInsert))
        return toBool(noInsert)

    def metricSetNoSwapping(self, noSwap: bool) -> None:
        """Set the flag indicating whether facet swapping should be turned off.

        Logically collective.

        Parameters
        ----------
        noSwap
            Flag indicating whether facet swapping should be turned off.

        See Also
        --------
        DMPlex.metricNoSwapping, DMPlex.metricSetNoInsertion
        DMPlex.metricSetNoMovement, DMPlex.metricSetNoSurf
        petsc.DMPlexMetricSetNoSwapping

        """
        cdef PetscBool bval = asBool(noSwap)
        CHKERR(DMPlexMetricSetNoSwapping(self.dm, bval))

    def metricNoSwapping(self) -> bool:
        """Return the flag indicating whether facet swapping is turned off.

        Not collective.

        See Also
        --------
        DMPlex.metricSetNoSwapping, DMPlex.metricNoInsertion
        DMPlex.metricNoMovement, DMPlex.metricNoSurf
        petsc.DMPlexMetricNoSwapping

        """
        cdef PetscBool noSwap = PETSC_FALSE
        CHKERR(DMPlexMetricNoSwapping(self.dm, &noSwap))
        return toBool(noSwap)

    def metricSetNoMovement(self, noMove: bool) -> None:
        """Set the flag indicating whether node movement should be turned off.

        Logically collective.

        Parameters
        ----------
        noMove
            Flag indicating whether node movement should be turned off.

        See Also
        --------
        DMPlex.metricNoMovement, DMPlex.metricSetNoInsertion
        DMPlex.metricSetNoSwapping, DMPlex.metricSetNoSurf
        petsc.DMPlexMetricSetNoMovement

        """
        cdef PetscBool bval = asBool(noMove)
        CHKERR(DMPlexMetricSetNoMovement(self.dm, bval))

    def metricNoMovement(self) -> bool:
        """Return the flag indicating whether node movement is turned off.

        Not collective.

        See Also
        --------
        DMPlex.metricSetNoMovement, DMPlex.metricNoInsertion
        DMPlex.metricNoSwapping, DMPlex.metricNoSurf
        petsc.DMPlexMetricNoMovement

        """
        cdef PetscBool noMove = PETSC_FALSE
        CHKERR(DMPlexMetricNoMovement(self.dm, &noMove))
        return toBool(noMove)

    def metricSetNoSurf(self, noSurf: bool) -> None:
        """Set the flag indicating whether surface modification should be turned off.

        Logically collective.

        Parameters
        ----------
        noSurf
            Flag indicating whether surface modification should be turned off.

        See Also
        --------
        DMPlex.metricNoSurf, DMPlex.metricSetNoMovement
        DMPlex.metricSetNoInsertion, DMPlex.metricSetNoSwapping
        petsc.DMPlexMetricSetNoSurf

        """
        cdef PetscBool bval = asBool(noSurf)
        CHKERR(DMPlexMetricSetNoSurf(self.dm, bval))

    def metricNoSurf(self) -> bool:
        """Return the flag indicating whether surface modification is turned off.

        Not collective.

        See Also
        --------
        DMPlex.metricSetNoSurf, DMPlex.metricNoMovement
        DMPlex.metricNoInsertion, DMPlex.metricNoSwapping
        petsc.DMPlexMetricNoSurf

        """
        cdef PetscBool noSurf = PETSC_FALSE
        CHKERR(DMPlexMetricNoSurf(self.dm, &noSurf))
        return toBool(noSurf)

    def metricSetVerbosity(self, verbosity: int) -> None:
        """Set the verbosity of the mesh adaptation package.

        Logically collective.

        Parameters
        ----------
        verbosity
            The verbosity, where -1 is silent and 10 is maximum.

        See Also
        --------
        DMPlex.metricGetVerbosity, DMPlex.metricSetNumIterations
        petsc.DMPlexMetricSetVerbosity

        """
        cdef PetscInt ival = asInt(verbosity)
        CHKERR(DMPlexMetricSetVerbosity(self.dm, ival))

    def metricGetVerbosity(self) -> int:
        """Return the verbosity of the mesh adaptation package.

        Not collective.

        Returns
        -------
        verbosity : int
            The verbosity, where -1 is silent and 10 is maximum.

        See Also
        --------
        DMPlex.metricSetVerbosity, DMPlex.metricGetNumIterations
        petsc.DMPlexMetricGetVerbosity

        """
        cdef PetscInt verbosity = 0
        CHKERR(DMPlexMetricGetVerbosity(self.dm, &verbosity))
        return toInt(verbosity)

    def metricSetNumIterations(self, numIter: int) -> None:
        """Set the number of parallel adaptation iterations.

        Logically collective.

        Parameters
        ----------
        numIter
            The number of parallel adaptation iterations.

        See Also
        --------
        DMPlex.metricSetVerbosity, DMPlex.metricGetNumIterations
        petsc.DMPlexMetricSetNumIterations

        """
        cdef PetscInt ival = asInt(numIter)
        CHKERR(DMPlexMetricSetNumIterations(self.dm, ival))

    def metricGetNumIterations(self) -> int:
        """Return the number of parallel adaptation iterations.

        Not collective.

        See Also
        --------
        DMPlex.metricSetNumIterations, DMPlex.metricGetVerbosity
        petsc.DMPlexMetricGetNumIterations

        """
        cdef PetscInt numIter = 0
        CHKERR(DMPlexMetricGetNumIterations(self.dm, &numIter))
        return toInt(numIter)

    def metricSetMinimumMagnitude(self, h_min: float) -> None:
        """Set the minimum tolerated metric magnitude.

        Logically collective.

        Parameters
        ----------
        h_min
            The minimum tolerated metric magnitude.

        See Also
        --------
        DMPlex.metricGetMinimumMagnitude, DMPlex.metricSetMaximumMagnitude
        petsc.DMPlexMetricSetMinimumMagnitude

        """
        cdef PetscReal rval = asReal(h_min)
        CHKERR(DMPlexMetricSetMinimumMagnitude(self.dm, rval))

    def metricGetMinimumMagnitude(self) -> float:
        """Return the minimum tolerated metric magnitude.

        Not collective.

        See Also
        --------
        DMPlex.metricSetMinimumMagnitude, DMPlex.metricGetMaximumMagnitude
        petsc.DMPlexMetricGetMinimumMagnitude

        """
        cdef PetscReal h_min = 0
        CHKERR(DMPlexMetricGetMinimumMagnitude(self.dm, &h_min))
        return toReal(h_min)

    def metricSetMaximumMagnitude(self, h_max: float) -> None:
        """Set the maximum tolerated metric magnitude.

        Logically collective.

        Parameters
        ----------
        h_max
            The maximum tolerated metric magnitude.

        See Also
        --------
        DMPlex.metricGetMaximumMagnitude, DMPlex.metricSetMinimumMagnitude
        petsc.DMPlexMetricSetMaximumMagnitude

        """
        cdef PetscReal rval = asReal(h_max)
        CHKERR(DMPlexMetricSetMaximumMagnitude(self.dm, rval))

    def metricGetMaximumMagnitude(self) -> float:
        """Return the maximum tolerated metric magnitude.

        Not collective.

        See Also
        --------
        DMPlex.metricSetMaximumMagnitude, DMPlex.metricGetMinimumMagnitude
        petsc.DMPlexMetricGetMaximumMagnitude

        """
        cdef PetscReal h_max = 0
        CHKERR(DMPlexMetricGetMaximumMagnitude(self.dm, &h_max))
        return toReal(h_max)

    def metricSetMaximumAnisotropy(self, a_max: float) -> None:
        """Set the maximum tolerated metric anisotropy.

        Logically collective.

        Parameters
        ----------
        a_max
            The maximum tolerated metric anisotropy.

        See Also
        --------
        DMPlex.metricGetMaximumAnisotropy, DMPlex.metricSetMaximumMagnitude
        petsc.DMPlexMetricSetMaximumAnisotropy

        """
        cdef PetscReal rval = asReal(a_max)
        CHKERR(DMPlexMetricSetMaximumAnisotropy(self.dm, rval))

    def metricGetMaximumAnisotropy(self) -> float:
        """Return the maximum tolerated metric anisotropy.

        Not collective.

        See Also
        --------
        DMPlex.metricSetMaximumAnisotropy, DMPlex.metricGetMaximumMagnitude
        petsc.DMPlexMetricGetMaximumAnisotropy

        """
        cdef PetscReal a_max = 0
        CHKERR(DMPlexMetricGetMaximumAnisotropy(self.dm, &a_max))
        return toReal(a_max)

    def metricSetTargetComplexity(self, targetComplexity: float) -> None:
        """Set the target metric complexity.

        Logically collective.

        Parameters
        ----------
        targetComplexity
            The target metric complexity.

        See Also
        --------
        DMPlex.metricGetTargetComplexity, DMPlex.metricSetNormalizationOrder
        petsc.DMPlexMetricSetTargetComplexity

        """
        cdef PetscReal rval = asReal(targetComplexity)
        CHKERR(DMPlexMetricSetTargetComplexity(self.dm, rval))

    def metricGetTargetComplexity(self) -> float:
        """Return the target metric complexity.

        Not collective.

        See Also
        --------
        DMPlex.metricSetTargetComplexity, DMPlex.metricGetNormalizationOrder
        petsc.DMPlexMetricGetTargetComplexity

        """
        cdef PetscReal targetComplexity = 0
        CHKERR(DMPlexMetricGetTargetComplexity(self.dm, &targetComplexity))
        return toReal(targetComplexity)

    def metricSetNormalizationOrder(self, p: float) -> None:
        """Set the order p for L-p normalization.

        Logically collective.

        Parameters
        ----------
        p
            The normalization order.

        See Also
        --------
        DMPlex.metricGetNormalizationOrder, DMPlex.metricSetTargetComplexity
        petsc.DMPlexMetricSetNormalizationOrder

        """
        cdef PetscReal rval = asReal(p)
        CHKERR(DMPlexMetricSetNormalizationOrder(self.dm, rval))

    def metricGetNormalizationOrder(self) -> float:
        """Return the order p for L-p normalization.

        Not collective.

        See Also
        --------
        DMPlex.metricSetNormalizationOrder, DMPlex.metricGetTargetComplexity
        petsc.DMPlexMetricGetNormalizationOrder

        """
        cdef PetscReal p = 0
        CHKERR(DMPlexMetricGetNormalizationOrder(self.dm, &p))
        return toReal(p)

    def metricSetGradationFactor(self, beta: float) -> None:
        """Set the metric gradation factor.

        Logically collective.

        Parameters
        ----------
        beta
            The metric gradation factor.

        See Also
        --------
        DMPlex.metricGetGradationFactor, DMPlex.metricSetHausdorffNumber
        petsc.DMPlexMetricSetGradationFactor

        """
        cdef PetscReal rval = asReal(beta)
        CHKERR(DMPlexMetricSetGradationFactor(self.dm, rval))

    def metricGetGradationFactor(self) -> float:
        """Return the metric gradation factor.

        Not collective.

        See Also
        --------
        DMPlex.metricSetGradationFactor, DMPlex.metricGetHausdorffNumber
        petsc.DMPlexMetricGetGradationFactor

        """
        cdef PetscReal beta = 0
        CHKERR(DMPlexMetricGetGradationFactor(self.dm, &beta))
        return toReal(beta)

    def metricSetHausdorffNumber(self, hausd: float) -> None:
        """Set the metric Hausdorff number.

        Logically collective.

        Parameters
        ----------
        hausd
            The metric Hausdorff number.

        See Also
        --------
        DMPlex.metricSetGradationFactor, DMPlex.metricGetHausdorffNumber
        petsc.DMPlexMetricSetHausdorffNumber

        """
        cdef PetscReal rval = asReal(hausd)
        CHKERR(DMPlexMetricSetHausdorffNumber(self.dm, rval))

    def metricGetHausdorffNumber(self) -> float:
        """Return the metric Hausdorff number.

        Not collective.

        See Also
        --------
        DMPlex.metricGetGradationFactor, DMPlex.metricSetHausdorffNumber
        petsc.DMPlexMetricGetHausdorffNumber

        """
        cdef PetscReal hausd = 0
        CHKERR(DMPlexMetricGetHausdorffNumber(self.dm, &hausd))
        return toReal(hausd)

    def metricCreate(self, field: int | None = 0) -> Vec:
        """Create a Riemannian metric field.

        Collective.

        Parameters
        ----------
        field
            The field number to use.

        See Also
        --------
        DMPlex.metricCreateUniform, DMPlex.metricCreateIsotropic
        petsc_options, petsc.DMPlexMetricCreate

        """
        cdef PetscInt ival = asInt(field)
        cdef Vec metric = Vec()
        CHKERR(DMPlexMetricCreate(self.dm, ival, &metric.vec))
        return metric

    def metricCreateUniform(self, alpha: float, field: int | None = 0) -> Vec:
        """Construct a uniform isotropic metric.

        Collective.

        Parameters
        ----------
        alpha
            Scaling parameter for the diagonal.
        field
            The field number to use.

        See Also
        --------
        DMPlex.metricCreate, DMPlex.metricCreateIsotropic
        petsc.DMPlexMetricCreateUniform

        """
        cdef PetscInt  ival = asInt(field)
        cdef PetscReal rval = asReal(alpha)
        cdef Vec metric = Vec()
        CHKERR(DMPlexMetricCreateUniform(self.dm, ival, rval, &metric.vec))
        return metric

    def metricCreateIsotropic(self, Vec indicator, field: int | None = 0) -> Vec:
        """Construct an isotropic metric from an error indicator.

        Collective.

        Parameters
        ----------
        indicator
            The error indicator.
        field
            The field number to use.

        See Also
        --------
        DMPlex.metricCreate, DMPlex.metricCreateUniform
        petsc.DMPlexMetricCreateIsotropic

        """
        cdef PetscInt  ival = asInt(field)
        cdef Vec metric = Vec()
        CHKERR(DMPlexMetricCreateIsotropic(self.dm, ival, indicator.vec, &metric.vec))
        return metric

    def metricDeterminantCreate(self, field: int | None = 0) -> tuple[Vec, DM]:
        """Create the determinant field for a Riemannian metric.

        Collective.

        Parameters
        ----------
        field
            The field number to use.

        Returns
        -------
        determinant : Vec
            The determinant field.
        dmDet : DM
            The corresponding DM

        See Also
        --------
        DMPlex.metricCreateUniform, DMPlex.metricCreateIsotropic
        DMPlex.metricCreate, petsc.DMPlexMetricDeterminantCreate

        """
        cdef PetscInt  ival = asInt(field)
        cdef Vec determinant = Vec()
        cdef DM dmDet = DM()
        CHKERR(DMPlexMetricDeterminantCreate(self.dm, ival, &determinant.vec, &dmDet.dm))
        return (determinant, dmDet)

    def metricEnforceSPD(self, Vec metric, Vec ometric, Vec determinant, restrictSizes: bool | None = False, restrictAnisotropy: bool | None = False) -> tuple[Vec, Vec]:
        """Enforce symmetric positive-definiteness of a metric.

        Collective.

        Parameters
        ----------
        metric
            The metric.
        ometric
            The output metric.
        determinant
            The output determinant.
        restrictSizes
            Flag indicating whether maximum/minimum magnitudes should be enforced.
        restrictAnisotropy
            Flag indicating whether maximum anisotropy should be enforced.

        Returns
        -------
        ometric : Vec
            The output metric.
        determinant : Vec
            The output determinant.

        See Also
        --------
        DMPlex.metricNormalize, DMPlex.metricIntersection2
        DMPlex.metricIntersection3, petsc_options, petsc.DMPlexMetricEnforceSPD

        """
        cdef PetscBool bval_rs = asBool(restrictSizes)
        cdef PetscBool bval_ra = asBool(restrictAnisotropy)
        CHKERR(DMPlexMetricEnforceSPD(self.dm, metric.vec, bval_rs, bval_ra, ometric.vec, determinant.vec))
        return (ometric, determinant)

    def metricNormalize(self, Vec metric, Vec ometric, Vec determinant, restrictSizes: bool | None = True, restrictAnisotropy: bool | None = True) -> tuple[Vec, Vec]:
        """Apply L-p normalization to a metric.

        Collective.

        Parameters
        ----------
        metric
            The metric.
        ometric
            The output metric.
        determinant
            The output determinant.
        restrictSizes
            Flag indicating whether maximum/minimum magnitudes should be enforced.
        restrictAnisotropy
            Flag indicating whether maximum anisotropy should be enforced.

        Returns
        -------
        ometric : Vec
            The output normalized metric.
        determinant : Vec
            The output determinant.

        See Also
        --------
        DMPlex.metricEnforceSPD, DMPlex.metricIntersection2
        DMPlex.metricIntersection3, petsc_options, petsc.DMPlexMetricNormalize

        """
        cdef PetscBool bval_rs = asBool(restrictSizes)
        cdef PetscBool bval_ra = asBool(restrictAnisotropy)
        CHKERR(DMPlexMetricNormalize(self.dm, metric.vec, bval_rs, bval_ra, ometric.vec, determinant.vec))
        return (ometric, determinant)

    def metricAverage2(self, Vec metric1, Vec metric2, Vec metricAvg) -> Vec:
        """Compute and return the unweighted average of two metrics.

        Collective.

        Parameters
        ----------
        metric1
            The first metric to be averaged.
        metric2
            The second metric to be averaged.
        metricAvg
            The output averaged metric.

        See Also
        --------
        DMPlex.metricAverage3, petsc.DMPlexMetricAverage2

        """
        CHKERR(DMPlexMetricAverage2(self.dm, metric1.vec, metric2.vec, metricAvg.vec))
        return metricAvg

    def metricAverage3(self, Vec metric1, Vec metric2, Vec metric3, Vec metricAvg) -> Vec:
        """Compute and return the unweighted average of three metrics.

        Collective.

        Parameters
        ----------
        metric1
            The first metric to be averaged.
        metric2
            The second metric to be averaged.
        metric3
            The third metric to be averaged.
        metricAvg
            The output averaged metric.

        See Also
        --------
        DMPlex.metricAverage2, petsc.DMPlexMetricAverage3

        """
        CHKERR(DMPlexMetricAverage3(self.dm, metric1.vec, metric2.vec, metric3.vec, metricAvg.vec))
        return metricAvg

    def metricIntersection2(self, Vec metric1, Vec metric2, Vec metricInt) -> Vec:
        """Compute and return the intersection of two metrics.

        Collective.

        Parameters
        ----------
        metric1
            The first metric to be intersected.
        metric2
            The second metric to be intersected.
        metricInt
            The output intersected metric.

        See Also
        --------
        DMPlex.metricIntersection3, petsc.DMPlexMetricIntersection2

        """
        CHKERR(DMPlexMetricIntersection2(self.dm, metric1.vec, metric2.vec, metricInt.vec))
        return metricInt

    def metricIntersection3(self, Vec metric1, Vec metric2, Vec metric3, Vec metricInt) -> Vec:
        """Compute the intersection of three metrics.

        Collective.

        Parameters
        ----------
        metric1
            The first metric to be intersected.
        metric2
            The second metric to be intersected.
        metric3
            The third metric to be intersected.
        metricInt
            The output intersected metric.

        See Also
        --------
        DMPlex.metricIntersection2, petsc.DMPlexMetricIntersection3

        """
        CHKERR(DMPlexMetricIntersection3(self.dm, metric1.vec, metric2.vec, metric3.vec, metricInt.vec))
        return metricInt

    def computeGradientClementInterpolant(self, Vec locX, Vec locC) -> Vec:
        """Return the L2 projection of the cellwise gradient of a function onto P1.

        Collective.

        Parameters
        ----------
        locX
            The coefficient vector of the function.
        locC
            The output `Vec` which holds the Clement interpolant of the gradient.

        See Also
        --------
        DM, DMPlex, petsc.DMPlexComputeGradientClementInterpolant

        """
        CHKERR(DMPlexComputeGradientClementInterpolant(self.dm, locX.vec, locC.vec))
        return locC

    # View

    def topologyView(self, Viewer viewer) -> None:
        """Save a `DMPlex` topology into a file.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` for saving.

        See Also
        --------
        DM, DMPlex, DM.view, DMPlex.coordinatesView, DMPlex.labelsView
        DMPlex.topologyLoad, Viewer, petsc.DMPlexTopologyView

        """
        CHKERR(DMPlexTopologyView(self.dm, viewer.vwr))

    def coordinatesView(self, Viewer viewer) -> None:
        """Save `DMPlex` coordinates into a file.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` for saving.

        See Also
        --------
        DM, DMPlex, DM.view, DMPlex.topologyView, DMPlex.labelsView
        DMPlex.coordinatesLoad, Viewer, petsc.DMPlexCoordinatesView

        """
        CHKERR(DMPlexCoordinatesView(self.dm, viewer.vwr))

    def labelsView(self, Viewer viewer) -> None:
        """Save `DMPlex` labels into a file.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` for saving.

        See Also
        --------
        DM, DMPlex, DM.view, DMPlex.topologyView, DMPlex.coordinatesView
        DMPlex.labelsLoad, Viewer, petsc.DMPlexLabelsView

        """
        CHKERR(DMPlexLabelsView(self.dm, viewer.vwr))

    def sectionView(self, Viewer viewer, DM sectiondm) -> None:
        """Save a section associated with a `DMPlex`.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` for saving.
        sectiondm
            The `DM` that contains the section to be saved.

        See Also
        --------
        DM, DMPlex, DM.view, DMPlex.topologyView, DMPlex.coordinatesView
        DMPlex.labelsView, DMPlex.globalVectorView, DMPlex.localVectorView
        DMPlex.sectionLoad, Viewer, petsc.DMPlexSectionView

        """
        CHKERR(DMPlexSectionView(self.dm, viewer.vwr, sectiondm.dm))

    def globalVectorView(self, Viewer viewer, DM sectiondm, Vec vec) -> None:
        """Save a global vector.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` to save data with.
        sectiondm
            The `DM` containing the global section on which ``vec``
            is defined; may be the same as this `DMPlex` object.
        vec
            The global vector to be saved.

        See Also
        --------
        DM, DMPlex, DMPlex.topologyView, DMPlex.sectionView
        DMPlex.localVectorView, DMPlex.globalVectorLoad
        DMPlex.localVectorLoad, petsc.DMPlexGlobalVectorView

        """
        CHKERR(DMPlexGlobalVectorView(self.dm, viewer.vwr, sectiondm.dm, vec.vec))

    def localVectorView(self, Viewer viewer, DM sectiondm, Vec vec) -> None:
        """Save a local vector.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` to save data with.
        sectiondm
            The `DM` that contains the local section on which ``vec`` is
            defined; may be the same as this `DMPlex` object.
        vec
            The local vector to be saved.

        See Also
        --------
        DM, DMPlex, DMPlex.topologyView, DMPlex.sectionView
        DMPlex.globalVectorView, DMPlex.globalVectorLoad
        DMPlex.localVectorLoad, petsc.DMPlexLocalVectorView

        """
        CHKERR(DMPlexLocalVectorView(self.dm, viewer.vwr, sectiondm.dm, vec.vec))

    # Load

    def topologyLoad(self, Viewer viewer) -> SF:
        """Load a topology into this `DMPlex` object.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` for the saved topology

        Returns
        -------
        sfxc : SF
            The `SF` that pushes points in ``[0, N)`` to the associated points
            in the loaded `DMPlex`, where ``N`` is the global number of points.

        See Also
        --------
        DM, DMPlex, DM.load, DMPlex.coordinatesLoad, DMPlex.labelsLoad
        DM.view, SF, Viewer, petsc.DMPlexTopologyLoad

        """
        cdef SF sf = SF()
        CHKERR(DMPlexTopologyLoad(self.dm, viewer.vwr, &sf.sf))
        return sf

    def coordinatesLoad(self, Viewer viewer, SF sfxc) -> None:
        """Load coordinates into this `DMPlex` object.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` for the saved coordinates.
        sfxc
            The `SF` returned by `topologyLoad`.

        See Also
        --------
        DM, DMPlex, DM.load, DMPlex.topologyLoad, DMPlex.labelsLoad, DM.view
        SF, Viewer, petsc.DMPlexCoordinatesLoad

        """
        CHKERR(DMPlexCoordinatesLoad(self.dm, viewer.vwr, sfxc.sf))

    def labelsLoad(self, Viewer viewer, SF sfxc) -> None:
        """Load labels into this `DMPlex` object.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` for the saved labels.
        sfxc
            The `SF` returned by `topologyLoad`.

        See Also
        --------
        DM, DMPlex, DM.load, DMPlex.topologyLoad, DMPlex.coordinatesLoad
        DM.view, SF, Viewer, petsc.DMPlexLabelsLoad

        """
        CHKERR(DMPlexLabelsLoad(self.dm, viewer.vwr, sfxc.sf))

    def sectionLoad(self, Viewer viewer, DM sectiondm, SF sfxc) -> tuple[SF, SF]:
        """Load section into a `DM`.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` that represents the on-disk section (``sectionA``).
        sectiondm
            The `DM` into which the on-disk section (``sectionA``) is migrated.
        sfxc
            The `SF` returned by `topologyLoad`.

        Returns
        -------
        gsf : SF
            The `SF` that migrates any on-disk `Vec` data associated with
            ``sectionA`` into a global `Vec` associated with the
            ``sectiondm``'s global section (`None` if not needed).
        lsf : SF
            The `SF` that migrates any on-disk `Vec` data associated with
            ``sectionA`` into a local `Vec` associated with the ``sectiondm``'s
            local section (`None` if not needed).

        See Also
        --------
        DM, DMPlex, DM.load, DMPlex.topologyLoad, DMPlex.coordinatesLoad
        DMPlex.labelsLoad, DMPlex.globalVectorLoad, DMPlex.localVectorLoad
        DMPlex.sectionView, SF, Viewer, petsc.DMPlexSectionLoad

        """
        cdef SF gsf = SF()
        cdef SF lsf = SF()
        CHKERR(DMPlexSectionLoad(self.dm, viewer.vwr, sectiondm.dm, sfxc.sf, &gsf.sf, &lsf.sf))
        return gsf, lsf

    def globalVectorLoad(self, Viewer viewer, DM sectiondm, SF sf, Vec vec) -> None:
        """Load on-disk vector data into a global vector.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` that represents the on-disk vector data.
        sectiondm
            The `DM` that contains the global section on which vec is defined.
        sf
            The `SF` that migrates the on-disk vector data into vec.
        vec
            The global vector to set values of.

        See Also
        --------
        DM, DMPlex, DMPlex.topologyLoad, DMPlex.sectionLoad
        DMPlex.localVectorLoad, DMPlex.globalVectorView
        DMPlex.localVectorView, SF, Viewer, petsc.DMPlexGlobalVectorLoad

        """
        CHKERR(DMPlexGlobalVectorLoad(self.dm, viewer.vwr, sectiondm.dm, sf.sf, vec.vec))

    def localVectorLoad(self, Viewer viewer, DM sectiondm, SF sf, Vec vec) -> None:
        """Load on-disk vector data into a local vector.

        Collective.

        Parameters
        ----------
        viewer
            The `Viewer` that represents the on-disk vector data.
        sectiondm
            The `DM` that contains the local section on which vec is defined.
        sf
            The `SF` that migrates the on-disk vector data into vec.
        vec
            The local vector to set values of.

        See Also
        --------
        DM, DMPlex, DMPlex.topologyLoad, DMPlex.sectionLoad
        DMPlex.globalVectorLoad, DMPlex.globalVectorView
        DMPlex.localVectorView, SF, Viewer, petsc.DMPlexLocalVectorLoad

        """
        CHKERR(DMPlexLocalVectorLoad(self.dm, viewer.vwr, sectiondm.dm, sf.sf, vec.vec))

    def createNaturalVec(self) -> Vec:
        """Return a natural vector.

        Collective.

        See Also
        --------
        petsc.DMPlexCreateNaturalVector

        """
        cdef Vec nv = Vec()
        CHKERR(DMPlexCreateNaturalVector(self.dm, &nv.vec))
        return nv

    def naturalToGlobalBegin(self, Vec nv, Vec gv) -> None:
        """Rearrange a `Vec` in the natural order to the Global order.

        Collective.

        See Also
        --------
        petsc.DMPlexNaturalToGlobalBegin

        """
        CHKERR(DMPlexNaturalToGlobalBegin(self.dm, nv.vec, gv.vec))

    def naturalToGlobalEnd(self, Vec nv, Vec gv) -> None:
        """Rearrange a `Vec` in the natural order to the Global order.

        Collective.

        See Also
        --------
        petsc.DMPlexNaturalToGlobalEnd

        """
        CHKERR(DMPlexNaturalToGlobalEnd(self.dm, nv.vec, gv.vec))

    def globalToNaturalBegin(self, Vec gv, Vec nv) -> None:
        """Rearrange a `Vec` in the Global order to the natural order.

        Collective.

        See Also
        --------
        petsc.DMPlexGlobalToNaturalBegin

        """
        CHKERR(DMPlexGlobalToNaturalBegin(self.dm, gv.vec, nv.vec))

    def globalToNaturalEnd(self, Vec gv, Vec nv) -> None:
        """Rearrange a `Vec` in the Global order to the natural order.

        Collective.

        See Also
        --------
        petsc.DMPlexGlobalToNaturalEnd

        """
        CHKERR(DMPlexGlobalToNaturalEnd(self.dm, gv.vec, nv.vec))

    def setMigrationSF(self, SF sf) -> None:
        """Set the `SF` for migrating from a parent `DM` into this `DM`.

        Not collective.

        See Also
        --------
        petsc.DMPlexSetMigrationSF

        """
        CHKERR(DMPlexSetMigrationSF(self.dm, sf.sf))

    def getMigrationSF(self) -> SF:
        """Get the `SF` for migrating from a parent `DM` into this `DM`.

        Not collective.

        See Also
        --------
        petsc.DMPlexGetMigrationSF

        """
        cdef SF sf = SF()
        CHKERR(DMPlexGetMigrationSF(self.dm, &sf.sf))
        CHKERR(PetscINCREF(sf.obj))
        return sf

    def createGlobalToNaturalSF(self, Section section, SF sfMigration) -> SF:
        """Create the `SF` for mapping Global `Vec` to the Natural `Vec`.

        Collective.

        See Also
        --------
        petsc.DMPlexCreateGlobalToNaturalSF

        """
        cdef SF sf = SF()
        CHKERR(DMPlexCreateGlobalToNaturalSF(self.dm, section.sec, sfMigration.sf, &sf.sf))
        return sf

    def migrateGlobalToNaturalSF(self, DM dmOld, SF sfNaturalOld, SF sfMigration) -> SF:
        """Create the `SF` for mapping Global `Vec` to the Natural `Vec` in the new `DM`.

        Collective.

        See Also
        --------
        petsc.DMPlexMigrateGlobalToNaturalSF

        """
        cdef SF sf = SF()
        CHKERR(DMPlexMigrateGlobalToNaturalSF(dmOld.dm, self.dm, sfNaturalOld.sf, sfMigration.sf, &sf.sf))
        return sf

# --------------------------------------------------------------------


class DMPlexTransformType(object):
    """Transformation types."""
    REFINEREGULAR = S_(DMPLEXREFINEREGULAR)
    REFINEALFELD = S_(DMPLEXREFINEALFELD)
    REFINEPOWELLSABIN = S_(DMPLEXREFINEPOWELLSABIN)
    REFINEBOUNDARYLAYER = S_(DMPLEXREFINEBOUNDARYLAYER)
    REFINESBR = S_(DMPLEXREFINESBR)
    REFINETOBOX = S_(DMPLEXREFINETOBOX)
    REFINETOSIMPLEX = S_(DMPLEXREFINETOSIMPLEX)
    REFINE1D = S_(DMPLEXREFINE1D)
    EXTRUDE = S_(DMPLEXEXTRUDETYPE)
    TRANSFORMFILTER = S_(DMPLEXTRANSFORMFILTER)


cdef class DMPlexTransform(Object):
    """Mesh transformations."""

    def __cinit__(self):
        self.obj = <PetscObject*> &self.tr
        self.tr  = NULL

    def apply(self, DM dm) -> DM:
        """Apply a mesh transformation.

        Collective.

        """
        # FIXME petsc.DMPlexTransformApply
        cdef DMPlex newdm = DMPlex()
        CHKERR(DMPlexTransformApply(self.tr, dm.dm, &newdm.dm))
        return newdm

    def create(self, comm: Comm | None = None) -> Self:
        """Create a mesh transformation.

        Collective.

        See Also
        --------
        petsc.DMPlexTransformCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDMPlexTransform newtr = NULL
        CHKERR(DMPlexTransformCreate(ccomm, &newtr))
        CHKERR(PetscCLEAR(self.obj))
        self.tr = newtr
        return self

    def destroy(self) -> Self:
        """Destroy a mesh transformation.

        Collective.

        See Also
        --------
        petsc.DMPlexTransformDestroy

        """
        CHKERR(DMPlexTransformDestroy(&self.tr))
        return self

    def getType(self) -> str:
        """Return the transformation type name.

        Not collective.

        See Also
        --------
        petsc.DMPlexTransformGetType

        """
        cdef PetscDMPlexTransformType cval = NULL
        CHKERR(DMPlexTransformGetType(self.tr, &cval))
        return bytes2str(cval)

    def setUp(self) -> Self:
        """Setup a mesh transformation.

        Collective.

        """
        # FIXME petsc.DMPlexTransformSetUp
        CHKERR(DMPlexTransformSetUp(self.tr))
        return self

    def setType(self, tr_type : DMPlexTransformType | str) -> None:
        """Set the transformation type.

        Collective.

        See Also
        --------
        petsc.DMPlexTransformSetType

        """
        cdef PetscDMPlexTransformType cval = NULL
        tr_type = str2bytes(tr_type, &cval)
        CHKERR(DMPlexTransformSetType(self.tr, cval))

    def setDM(self, DM dm) -> None:
        """Set the `DM` for the transformation.

        Logically collective.

        """
        # FIXME petsc.DMPlexTransformSetDM
        CHKERR(DMPlexTransformSetDM(self.tr, dm.dm))

    def setFromOptions(self) -> None:
        """Configure the transformation from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.DMPlexTransformSetFromOptions

        """
        CHKERR(DMPlexTransformSetFromOptions(self.tr))

    def view(self, Viewer viewer=None) -> None:
        """View the mesh transformation.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        Viewer, petsc.DMPlexTransformView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR(DMPlexTransformView(self.tr, vwr))
