# --------------------------------------------------------------------

class DMSwarmType(object):
    BASIC = DMSWARM_BASIC
    PIC = DMSWARM_PIC

class DMSwarmMigrateType(object):
    MIGRATE_BASIC = DMSWARM_MIGRATE_BASIC
    MIGRATE_DMCELLNSCATTER = DMSWARM_MIGRATE_DMCELLNSCATTER
    MIGRATE_DMCELLEXACT = DMSWARM_MIGRATE_DMCELLEXACT
    MIGRATE_USER = DMSWARM_MIGRATE_USER

class DMSwarmCollectType(object):
    COLLECT_BASIC = DMSWARM_COLLECT_BASIC
    COLLECT_DMDABOUNDINGBOX = DMSWARM_COLLECT_DMDABOUNDINGBOX
    COLLECT_GENERAL = DMSWARM_COLLECT_GENERAL
    COLLECT_USER = DMSWARM_COLLECT_USER

class DMSwarmPICLayoutType(object):
    LAYOUT_REGULAR = DMSWARMPIC_LAYOUT_REGULAR
    LAYOUT_GAUSS = DMSWARMPIC_LAYOUT_GAUSS
    LAYOUT_SUBDIVISION = DMSWARMPIC_LAYOUT_SUBDIVISION


cdef class DMSwarm(DM):
    """A `DM` object used to represent arrays of data (fields) of arbitrary type."""
    Type = DMSwarmType
    MigrateType = DMSwarmMigrateType
    CollectType = DMSwarmCollectType
    PICLayoutType = DMSwarmPICLayoutType

    def create(self, comm: Comm | None = None) -> Self:
        """Create an empty DM object and set its type to `DM.Type.SWARM`.

        Collective.

        DMs are the abstract objects in PETSc that mediate between meshes and
        discretizations and the algebraic solvers, time integrators, and
        optimization algorithms.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.DMCreate, petsc.DMSetType

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR( DMCreate(ccomm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        CHKERR( DMSetType(self.dm, DMSWARM) )
        return self

    def createGlobalVectorFromField(self, fieldname: str) -> Vec:
        """Create a global `Vec` object associated with a given field.

        Collective.

        The vector must be returned to the `DMSwarm` using a matching call to
        `destroyGlobalVectorFromField`.

        Parameters
        ----------
        fieldname
            The textual name given to a registered field.

        See Also
        --------
        destroyGlobalVectorFromField, petsc.DMSwarmCreateGlobalVectorFromField

        """
        cdef const char *cfieldname = NULL
        cdef Vec vg = Vec()
        fieldname = str2bytes(fieldname, &cfieldname)
        CHKERR( DMSwarmCreateGlobalVectorFromField(self.dm, cfieldname, &vg.vec) )
        return vg

    def destroyGlobalVectorFromField(self, fieldname: str) -> None:
        """Destroy the global `Vec` object associated with a given field.

        Collective.

        Parameters
        ----------
        fieldname
            The textual name given to a registered field.

        See Also
        --------
        createGlobalVectorFromField, petsc.DMSwarmDestroyGlobalVectorFromField

        """
        cdef const char *cfieldname = NULL
        cdef PetscVec vec = NULL
        fieldname = str2bytes(fieldname, &cfieldname)
        CHKERR( DMSwarmDestroyGlobalVectorFromField(self.dm, cfieldname, &vec) )

    def createLocalVectorFromField(self, fieldname: str) -> Vec:
        """Create a local `Vec` object associated with a given field.

        Collective.

        The vector must be returned to the `DMSwarm` using a matching call
        to `destroyLocalVectorFromField`.

        Parameters
        ----------
        fieldname
            The textual name given to a registered field.

        See Also
        --------
        destroyLocalVectorFromField, petsc.DMSwarmCreateLocalVectorFromField

        """
        cdef const char *cfieldname = NULL
        cdef Vec vl = Vec()
        fieldname = str2bytes(fieldname, &cfieldname)
        CHKERR( DMSwarmCreateLocalVectorFromField(self.dm, cfieldname, &vl.vec) )
        return vl

    def destroyLocalVectorFromField(self, fieldname: str) -> None:
        """Destroy the local `Vec` object associated with a given field.

        Collective.

        Parameters
        ----------
        fieldname
            The textual name given to a registered field.

        See Also
        --------
        createLocalVectorFromField, petsc.DMSwarmDestroyLocalVectorFromField

        """
        cdef const char *cfieldname = NULL
        cdef PetscVec vec
        fieldname = str2bytes(fieldname, &cfieldname)
        CHKERR( DMSwarmDestroyLocalVectorFromField(self.dm, cfieldname, &vec) )

    def initializeFieldRegister(self) -> None:
        """Initiate the registration of fields to a `DMSwarm`.

        Collective.

        After all fields have been registered, you must call `finalizeFieldRegister`.

        See Also
        --------
        finalizeFieldRegister, petsc.DMSwarmInitializeFieldRegister

        """
        CHKERR( DMSwarmInitializeFieldRegister(self.dm) )

    def finalizeFieldRegister(self) -> None:
        """Finalize the registration of fields to a `DMSwarm`.

        Collective.

        See Also
        --------
        initializeFieldRegister, petsc.DMSwarmFinalizeFieldRegister

        """
        CHKERR( DMSwarmFinalizeFieldRegister(self.dm) )

    def setLocalSizes(self, nlocal: int, buffer: int) -> Self:
        """Set the length of all registered fields on the `DMSwarm`.

        Not collective.

        Parameters
        ----------
        nlocal
            The length of each registered field.
        buffer
            The length of the buffer used for efficient dynamic resizing.

        See Also
        --------
        petsc.DMSwarmSetLocalSizes

        """
        cdef PetscInt cnlocal = asInt(nlocal)
        cdef PetscInt cbuffer = asInt(buffer)
        CHKERR( DMSwarmSetLocalSizes(self.dm, cnlocal, cbuffer) )
        return self

    def registerField(self, fieldname: str, blocksize: int, dtype: dtype = ScalarType) -> None:
        """Register a field to a `DMSwarm` with a native PETSc data type.

        Collective.

        Parameters
        ----------
        fieldname
            The textual name to identify this field.
        blocksize
            The number of each data type.
        dtype
            A valid PETSc data type.

        See Also
        --------
        petsc.DMSwarmRegisterPetscDatatypeField

        """
        cdef const char *cfieldname = NULL
        cdef PetscInt cblocksize = asInt(blocksize)
        cdef PetscDataType ctype  = PETSC_DATATYPE_UNKNOWN
        if dtype == IntType:     ctype = PETSC_INT
        if dtype == RealType:    ctype = PETSC_REAL
        if dtype == ScalarType:  ctype = PETSC_SCALAR
        if dtype == ComplexType: ctype = PETSC_COMPLEX
        assert ctype != PETSC_DATATYPE_UNKNOWN
        fieldname = str2bytes(fieldname, &cfieldname)
        CHKERR( DMSwarmRegisterPetscDatatypeField(self.dm, cfieldname, cblocksize, ctype) )

    def getField(self, fieldname: str) -> Sequence[int | float | complex]:
        """Return arrays storing all entries associated with a field.

        Not collective.

        The returned array contains underlying values of the field.

        The array must be returned to the `DMSwarm` using a matching call to
        `restoreField`.

        Parameters
        ----------
        fieldname
            The textual name to identify this field.

        Returns
        -------
        `numpy.ndarray`
            The type of the entries in the array will match the type of the
            field.

        See Also
        --------
        restoreField, petsc.DMSwarmGetField

        """
        cdef const char *cfieldname = NULL
        cdef PetscInt blocksize = 0
        cdef PetscDataType ctype = PETSC_DATATYPE_UNKNOWN
        cdef PetscReal *data = NULL
        cdef PetscInt nlocal = 0
        fieldname = str2bytes(fieldname, &cfieldname)
        CHKERR( DMSwarmGetField(self.dm, cfieldname, &blocksize, &ctype, <void**> &data) )
        CHKERR( DMSwarmGetLocalSize(self.dm, &nlocal) )
        cdef int typenum = -1
        if ctype == PETSC_INT:     typenum = NPY_PETSC_INT
        if ctype == PETSC_REAL:    typenum = NPY_PETSC_REAL
        if ctype == PETSC_SCALAR:  typenum = NPY_PETSC_SCALAR
        if ctype == PETSC_COMPLEX: typenum = NPY_PETSC_COMPLEX
        assert typenum != -1
        cdef npy_intp s = <npy_intp> nlocal * blocksize
        return <object> PyArray_SimpleNewFromData(1, &s, typenum, data)

    def restoreField(self, fieldname: str) -> None:
        """Restore accesses associated with a registered field.

        Not collective.

        Parameters
        ----------
        fieldname
            The textual name to identify this field.

        See Also
        --------
        getField, petsc.DMSwarmRestoreField

        """
        cdef const char *cfieldname = NULL
        cdef PetscInt blocksize = 0
        cdef PetscDataType ctype = PETSC_DATATYPE_UNKNOWN
        fieldname = str2bytes(fieldname, &cfieldname)
        CHKERR( DMSwarmRestoreField(self.dm, cfieldname, &blocksize, &ctype, <void**> 0) )

    def vectorDefineField(self, fieldname: str) -> None:
        """Set the field from which to define a `Vec` object.

        Collective.

        The field will be used when `DM.createLocalVec`, or
        `DM.createGlobalVec` is called.

        Parameters
        ----------
        fieldname
            The textual name given to a registered field.

        See Also
        --------
        petsc.DMSwarmVectorDefineField

        """
        cdef const char *cval = NULL
        fieldname = str2bytes(fieldname, &cval)
        CHKERR( DMSwarmVectorDefineField(self.dm, cval) )

    def addPoint(self) -> None:
        """Add space for one new point in the `DMSwarm`.

        Not collective.

        See Also
        --------
        petsc.DMSwarmAddPoint

        """
        CHKERR( DMSwarmAddPoint(self.dm) )

    def addNPoints(self, npoints: int) -> None:
        """Add space for a number of new points in the `DMSwarm`.

        Not collective.

        Parameters
        ----------
        npoints
            The number of new points to add.

        See Also
        --------
        petsc.DMSwarmAddNPoints

        """
        cdef PetscInt cnpoints = asInt(npoints)
        CHKERR( DMSwarmAddNPoints(self.dm, cnpoints) )

    def removePoint(self) -> None:
        """Remove the last point from the `DMSwarm`.

        Not collective.

        See Also
        --------
        petsc.DMSwarmRemovePoint

        """
        CHKERR( DMSwarmRemovePoint(self.dm) )

    def removePointAtIndex(self, index: int) -> None:
        """Remove a specific point from the `DMSwarm`

        Not collective.

        Parameters
        ----------
        index
            Index of point to remove

        See Also
        --------
        petsc.DMSwarmRemovePointAtIndex

        """
        cdef PetscInt cindex = asInt(index)
        CHKERR( DMSwarmRemovePointAtIndex(self.dm, cindex) )

    def copyPoint(self, pi: int, pj: int) -> None:
        """Copy point pi to point pj in the `DMSwarm`.

        Not collective.

        Parameters
        ----------
        pi
            The index of the point to copy (source).
        pj
            The point index where the copy should be located (destination).

        See Also
        --------
        petsc.DMSwarmCopyPoint

        """
        cdef PetscInt cpi = asInt(pi)
        cdef PetscInt cpj = asInt(pj)
        CHKERR( DMSwarmCopyPoint(self.dm, cpi, cpj) )

    def getLocalSize(self) -> int:
        """Return the local length of fields registered.

        Not collective.

        See Also
        --------
        petsc.DMSwarmGetLocalSize

        """
        cdef PetscInt size = asInt(0)
        CHKERR( DMSwarmGetLocalSize(self.dm, &size) )
        return toInt(size)

    def getSize(self) -> int:
        """Return the total length of fields registered.

        Collective.

        See Also
        --------
        petsc.DMSwarmGetSize

        """
        cdef PetscInt size = asInt(0)
        CHKERR( DMSwarmGetSize(self.dm, &size) )
        return toInt(size)

    def migrate(self, remove_sent_points: bool = False) -> None:
        """Relocate points defined in the `DMSwarm` to other MPI ranks.

        Collective.

        Parameters
        ----------
        remove_sent_points
            Flag indicating if sent points should be removed from the current
            MPI rank.

        See Also
        --------
        petsc.DMSwarmMigrate

        """
        cdef PetscBool remove_pts = asBool(remove_sent_points)
        CHKERR( DMSwarmMigrate(self.dm, remove_pts) )

    def collectViewCreate(self) -> None:
        """Apply a collection method and gather points in neighbor ranks.

        Collective.

        See Also
        --------
        collectViewDestroy, petsc.DMSwarmCollectViewCreate

        """
        CHKERR( DMSwarmCollectViewCreate(self.dm) )

    def collectViewDestroy(self) -> None:
        """Reset the `DMSwarm` to the size prior to calling `collectViewCreate`.

        Collective.

        See Also
        --------
        collectViewCreate, petsc.DMSwarmCollectViewDestroy

        """
        CHKERR( DMSwarmCollectViewDestroy(self.dm) )

    def setCellDM(self, DM dm) -> None:
        """Attach a `DM` to a `DMSwarm`.

        Collective.

        Parameters
        ----------
        dm
            The `DM` to attach to the `DMSwarm`.

        See Also
        --------
        getCellDM, petsc.DMSwarmSetCellDM

        """
        CHKERR( DMSwarmSetCellDM(self.dm, dm.dm) )

    def getCellDM(self) -> DM:
        """Return `DM` cell attached to `DMSwarm`.

        Collective.

        See Also
        --------
        setCellDM, petsc.DMSwarmGetCellDM

        """
        cdef PetscDM newdm = NULL
        CHKERR( DMSwarmGetCellDM(self.dm, &newdm) )
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        PetscINCREF(dm.obj)
        return dm

    def setType(self, dmswarm_type: Type | str) -> None:
        """Set particular flavor of `DMSwarm`.

        Collective.

        Parameters
        ----------
        dmswarm_type
            The `DMSwarm` type.

        See Also
        --------
        petsc.DMSwarmSetType

        """
        cdef PetscDMSwarmType cval = dmswarm_type
        CHKERR( DMSwarmSetType(self.dm, cval) )

    def setPointsUniformCoordinates(
        self,
        min: Sequence[float],
        max: Sequence[float],
        npoints: Sequence[int],
        mode: InsertMode | None = None,
    ) -> Self:
        """Set point coordinates in a `DMSwarm` on a regular (ijk) grid.

        Collective.

        Parameters
        ----------
        min
            Minimum coordinate values in the x, y, z directions (array of
            length ``dim``).
        max
            Maximum coordinate values in the x, y, z directions (array of
            length ``dim``).
        npoints
            Number of points in each spatial direction (array of length ``dim``).
        mode
            Indicates whether to append points to the swarm (`InsertMode.ADD`),
            or override existing points (`InsertMode.INSERT`).

        See Also
        --------
        petsc.DMSwarmSetPointsUniformCoordinates

        """
        cdef PetscInt dim = asInt(0)
        CHKERR( DMGetDimension(self.dm, &dim) )
        cdef PetscReal cmin[3]
        cmin[0] = cmin[1] = cmin[2] = asReal(0.)
        for i from 0 <= i < dim: cmin[i] = min[i]
        cdef PetscReal cmax[3]
        cmax[0] = cmax[1] = cmax[2] = asReal(0.)
        for i from 0 <= i < dim: cmax[i] = max[i]
        cdef PetscInt cnpoints[3]
        cnpoints[0] = cnpoints[1] = cnpoints[2] = asInt(0)
        for i from 0 <= i < dim: cnpoints[i] = npoints[i]
        cdef PetscInsertMode cmode = insertmode(mode)
        CHKERR( DMSwarmSetPointsUniformCoordinates(self.dm, cmin, cmax, cnpoints, cmode) )
        return self

    def setPointCoordinates(
        self,
        coordinates: Sequence[float],
        redundant: bool = False,
        mode: InsertMode | None = None
    ) -> None:
        """Set point coordinates in a `DMSwarm` from a user-defined list.

        Collective.

        Parameters
        ----------
        coordinates
            The coordinate values.
        redundant
            If set to `True`, it is assumed that coordinates are only valid on
            rank 0 and should be broadcast to other ranks.
        mode
            Indicates whether to append points to the swarm (`InsertMode.ADD`),
            or override existing points (`InsertMode.INSERT`).

        See Also
        --------
        petsc.DMSwarmSetPointCoordinates

        """
        cdef ndarray xyz = iarray(coordinates, NPY_PETSC_REAL)
        if PyArray_ISFORTRAN(xyz): xyz = PyArray_Copy(xyz)
        if PyArray_NDIM(xyz) != 2: raise ValueError(
            ("coordinates must have two dimensions: "
             "coordinates.ndim=%d") % (PyArray_NDIM(xyz)) )
        cdef PetscInt cnpoints = <PetscInt> PyArray_DIM(xyz, 0)
        cdef PetscBool credundant = asBool(redundant)
        cdef PetscInsertMode cmode = insertmode(mode)
        cdef PetscReal *coords = <PetscReal*> PyArray_DATA(xyz)
        CHKERR( DMSwarmSetPointCoordinates(self.dm, cnpoints, coords, credundant, cmode) )

    def insertPointUsingCellDM(self, layoutType: PICLayoutType, fill_param: int) -> None:
        """Insert point coordinates within each cell.

        Not collective.

        Parameters
        ----------
        layout_type
            Method used to fill each cell with the cell DM.
        fill_param
            Parameter controlling how many points per cell are added (the
            meaning of this parameter is dependent on the layout type).

        See Also
        --------
        petsc.DMSwarmInsertPointsUsingCellDM

        """
        cdef PetscDMSwarmPICLayoutType clayoutType = layoutType
        cdef PetscInt cfill_param = asInt(fill_param)
        CHKERR( DMSwarmInsertPointsUsingCellDM(self.dm, clayoutType, cfill_param) )

    def setPointCoordinatesCellwise(self, coordinates: Sequence[float]) -> None:
        """Insert point coordinates within each cell.

        Not collective.

        Point coordinates are defined over the reference cell.

        Parameters
        ----------
        coordinates
            The coordinates (defined in the local coordinate system for each
            cell) to insert.

        See Also
        --------
        petsc.DMSwarmSetPointCoordinatesCellwise

        """
        cdef ndarray xyz = iarray(coordinates, NPY_PETSC_REAL)
        if PyArray_ISFORTRAN(xyz): xyz = PyArray_Copy(xyz)
        if PyArray_NDIM(xyz) != 2: raise ValueError(
            ("coordinates must have two dimensions: "
             "coordinates.ndim=%d") % (PyArray_NDIM(xyz)) )
        cdef PetscInt cnpoints = <PetscInt> PyArray_DIM(xyz, 0)
        cdef PetscReal *coords = <PetscReal*> PyArray_DATA(xyz)
        CHKERR( DMSwarmSetPointCoordinatesCellwise(self.dm, cnpoints, coords) )

    def viewFieldsXDMF(self, filename: str, fieldnames: Sequence[str]) -> None:
        """Write a selection of `DMSwarm` fields to an XDMF3 file.

        Collective.

        Parameters
        ----------
        filename
            The file name of the XDMF file (must have the extension .xmf).
        fieldnames
            Array containing the textual names of fields to write.

        See Also
        --------
        petsc.DMSwarmViewFieldsXDMF

        """
        cdef const char *cval = NULL
        cdef const char *cfilename = NULL
        filename = str2bytes(filename, &cfilename)
        cdef PetscInt cnfields = <PetscInt> len(fieldnames)
        cdef const char** cfieldnames = NULL
        cdef object tmp = oarray_p(empty_p(cnfields), NULL, <void**>&cfieldnames)
        fieldnames = list(fieldnames)
        for i from 0 <= i < cnfields:
            fieldnames[i] = str2bytes(fieldnames[i], &cval)
            cfieldnames[i] = cval
        CHKERR( DMSwarmViewFieldsXDMF(self.dm, cfilename, cnfields, cfieldnames ) )

    def viewXDMF(self, filename: str) -> None:
        """Write this `DMSwarm` fields to an XDMF3 file.

        Collective.

        Parameters
        ----------
        filename
            The file name of the XDMF file (must have the extension .xmf).

        See Also
        --------
        petsc.DMSwarmViewXDMF

        """
        cdef const char *cval = NULL
        filename = str2bytes(filename, &cval)
        CHKERR( DMSwarmViewXDMF(self.dm, cval) )

    def sortGetAccess(self) -> None:
        """Setup up a `DMSwarm` point sort context.

        Not collective.

        The point sort context is used for efficient traversal of points within
        a cell.

        You must call `sortRestoreAccess` when you no longer need access to the
        sort context.

        See Also
        --------
        sortRestoreAccess, petsc.DMSwarmSortGetAccess

        """
        CHKERR( DMSwarmSortGetAccess(self.dm) )

    def sortRestoreAccess(self) -> None:
        """Invalidate the `DMSwarm` point sorting context.

        Not collective.

        See Also
        --------
        sortGetAccess, petsc.DMSwarmSortRestoreAccess

        """
        CHKERR( DMSwarmSortRestoreAccess(self.dm) )

    def sortGetPointsPerCell(self, e: int) -> list[int]:
        """Create an array of point indices for all points in a cell.

        Not collective.

        Parameters
        ----------
        e
            The index of the cell.

        See Also
        --------
        petsc.DMSwarmSortGetPointsPerCell

        """
        cdef PetscInt ce = asInt(e)
        cdef PetscInt cnpoints = asInt(0)
        cdef PetscInt *cpidlist = NULL
        cdef list pidlist = []
        CHKERR( DMSwarmSortGetPointsPerCell(self.dm, ce, &cnpoints, &cpidlist) )
        npoints = asInt(cnpoints)
        for i from 0 <= i < npoints: pidlist.append(asInt(cpidlist[i]))
        return pidlist

    def sortGetNumberOfPointsPerCell(self, e: int) -> int:
        """Return the number of points in a cell.

        Not collective.

        Parameters
        ----------
        e
            The index of the cell.

        See Also
        --------
        petsc.DMSwarmSortGetNumberOfPointsPerCell

        """
        cdef PetscInt ce = asInt(e)
        cdef PetscInt npoints = asInt(0)
        CHKERR( DMSwarmSortGetNumberOfPointsPerCell(self.dm, ce, &npoints) )
        return toInt(npoints)

    def sortGetIsValid(self) -> bool:
        """Return whether the sort context is up-to-date.

        Not collective.

        Returns the flag associated with a `DMSwarm` point sorting context.

        See Also
        --------
        petsc.DMSwarmSortGetIsValid

        """
        cdef PetscBool isValid = asBool(False)
        CHKERR( DMSwarmSortGetIsValid(self.dm, &isValid) )
        return toBool(isValid)

    def sortGetSizes(self) -> tuple[int, int]:
        """Return the sizes associated with a `DMSwarm` point sorting context.

        Not collective.

        Returns
        -------
        ncells : int
            Number of cells within the sort context.
        npoints : int
            Number of points used to create the sort context.

        See Also
        --------
        petsc.DMSwarmSortGetSizes

        """
        cdef PetscInt ncells = asInt(0)
        cdef PetscInt npoints = asInt(0)
        CHKERR( DMSwarmSortGetSizes(self.dm, &ncells, &npoints) )
        return (toInt(ncells), toInt(npoints))

    def projectFields(self, fieldnames: Sequence[str], reuse: bool = False) -> list[Vec]:
        """Project a set of `DMSwarm` fields onto the cell `DM`.

        Collective.

        Parameters
        ----------
        fieldnames
            The textual names of the swarm fields to project.
        reuse
            Flag indicating whether the array and contents of fields should be
            reused or internally allocated.

        See Also
        --------
        petsc.DMSwarmProjectFields

        """
        cdef PetscBool creuse = asBool(reuse)
        cdef const char *cval = NULL
        cdef PetscInt cnfields = <PetscInt> len(fieldnames)
        cdef const char** cfieldnames = NULL
        cdef object tmp = oarray_p(empty_p(cnfields), NULL, <void**>&cfieldnames)
        cdef PetscVec *cfieldvecs
        fieldnames = list(fieldnames)
        for i from 0 <= i < cnfields:
            fieldnames[i] = str2bytes(fieldnames[i], &cval)
            cfieldnames[i] = cval
        CHKERR( DMSwarmProjectFields(self.dm, cnfields, cfieldnames, &cfieldvecs, creuse) )
        cdef list fieldvecs = []
        for i from 0 <= i < cnfields:
            newVec = Vec()
            newVec.vec = cfieldvecs[i]
            fieldvecs.append(newVec)
        return fieldvecs


del DMSwarmType
del DMSwarmMigrateType
del DMSwarmCollectType
del DMSwarmPICLayoutType
