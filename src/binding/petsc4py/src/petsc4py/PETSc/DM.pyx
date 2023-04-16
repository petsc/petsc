# --------------------------------------------------------------------

class DMType(object):
    """`DM` types."""
    DA        = S_(DMDA_type)
    COMPOSITE = S_(DMCOMPOSITE)
    SLICED    = S_(DMSLICED)
    SHELL     = S_(DMSHELL)
    PLEX      = S_(DMPLEX)
    REDUNDANT = S_(DMREDUNDANT)
    PATCH     = S_(DMPATCH)
    MOAB      = S_(DMMOAB)
    NETWORK   = S_(DMNETWORK)
    FOREST    = S_(DMFOREST)
    P4EST     = S_(DMP4EST)
    P8EST     = S_(DMP8EST)
    SWARM     = S_(DMSWARM)
    PRODUCT   = S_(DMPRODUCT)
    STAG      = S_(DMSTAG)

class DMBoundaryType(object):
    """`DM` Boundary types."""
    NONE     = DM_BOUNDARY_NONE
    GHOSTED  = DM_BOUNDARY_GHOSTED
    MIRROR   = DM_BOUNDARY_MIRROR
    PERIODIC = DM_BOUNDARY_PERIODIC
    TWIST    = DM_BOUNDARY_TWIST

class DMPolytopeType(object):
    POINT              = DM_POLYTOPE_POINT
    SEGMENT            = DM_POLYTOPE_SEGMENT
    POINT_PRISM_TENSOR = DM_POLYTOPE_POINT_PRISM_TENSOR
    TRIANGLE           = DM_POLYTOPE_TRIANGLE
    QUADRILATERAL      = DM_POLYTOPE_QUADRILATERAL
    SEG_PRISM_TENSOR   = DM_POLYTOPE_SEG_PRISM_TENSOR
    TETRAHEDRON        = DM_POLYTOPE_TETRAHEDRON
    HEXAHEDRON         = DM_POLYTOPE_HEXAHEDRON
    TRI_PRISM          = DM_POLYTOPE_TRI_PRISM
    TRI_PRISM_TENSOR   = DM_POLYTOPE_TRI_PRISM_TENSOR
    QUAD_PRISM_TENSOR  = DM_POLYTOPE_QUAD_PRISM_TENSOR
    PYRAMID            = DM_POLYTOPE_PYRAMID
    FV_GHOST           = DM_POLYTOPE_FV_GHOST
    INTERIOR_GHOST     = DM_POLYTOPE_INTERIOR_GHOST
    UNKNOWN            = DM_POLYTOPE_UNKNOWN

# --------------------------------------------------------------------

cdef class DM(Object):
    """An object describing a computational grid or mesh.

    DM is documented in `the PETSc manual <petsc:chapter_dmbase>`.

    """

    Type         = DMType
    BoundaryType = DMBoundaryType
    PolytopeType = DMPolytopeType

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.dm
        self.dm  = NULL

    def view(self, Viewer viewer=None) -> None:
        """View the `DM`.

        Collective.

        Parameters
        ----------
        viewer
            The DM viewer.

        See Also
        --------
        petsc.DMView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( DMView(self.dm, vwr) )

    def load(self, Viewer viewer) -> Self:
        """Return a `DM` stored in binary.

        Collective.

        Parameters
        ----------
        viewer
            Viewer used to display the `DM`, either `Viewer.Type.BINARY` or `Viewer.Type.HDF5`.

        Notes
        -----
        When using `Viewer.Type.HDF5` format, one can save multiple `DMPlex` meshes in a single HDF5 files.
        This in turn requires one to name the `DMPlex` object with `Object.setName` before saving it with `DM.view` and before loading it with `DM.load` for identification of the mesh object.

        See Also
        --------
        DM.view, DM.load, Object.setName, petsc.DMLoad

        """
        CHKERR( DMLoad(self.dm, viewer.vwr) )
        return self

    def destroy(self) -> Self:
        """Destroy the object.

        Collective.

        See Also
        --------
        petsc.DMDestroy

        """
        CHKERR( DMDestroy(&self.dm) )
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Return an empty `DM`.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.DMCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR( DMCreate(ccomm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def clone(self) -> DM:
        """Return the cloned `DM` .

        Collective.

        See Also
        --------
        petsc.DMClone

        """
        cdef DM dm = type(self)()
        CHKERR( DMClone(self.dm, &dm.dm) )
        return dm

    def setType(self, dm_type: DM.Type | str) -> None:
        """Build a `DM`.

        Collective.

        Parameters
        ----------
        dm_type
            The type of `DM`.

        Notes
        -----
        `DM` types are available in `DM.Type` class.

        See Also
        --------
        DM.Type, petsc.DMSetType

        """
        cdef PetscDMType cval = NULL
        dm_type = str2bytes(dm_type, &cval)
        CHKERR( DMSetType(self.dm, cval) )

    def getType(self) -> str:
        """Return the `DM` type name.

        Not collective.

        See Also
        --------
        petsc.DMGetType

        """
        cdef PetscDMType cval = NULL
        CHKERR( DMGetType(self.dm, &cval) )
        return bytes2str(cval)

    def getDimension(self) -> int:
        """Return the topological dimension of the `DM`.

        Not collective.

        See Also
        --------
        petsc.DMGetDimension

        """
        cdef PetscInt dim = 0
        CHKERR( DMGetDimension(self.dm, &dim) )
        return toInt(dim)

    def setDimension(self, dim: int) -> None:
        """Set the topological dimension of the `DM`.

        Collective.

        Parameters
        ----------
        dim
            Topological dimension.

        See Also
        --------
        petsc.DMSetDimension

        """
        cdef PetscInt cdim = asInt(dim)
        CHKERR( DMSetDimension(self.dm, cdim) )

    def getCoordinateDim(self) -> int:
        """Return the dimension of embedding space for coordinates values.

        Not collective.

        See Also
        --------
        petsc.DMGetCoordinateDim

        """
        cdef PetscInt dim = 0
        CHKERR( DMGetCoordinateDim(self.dm, &dim) )
        return toInt(dim)

    def setCoordinateDim(self, dim: int) -> None:
        """Set the dimension of embedding space for coordinates values.

        Not collective.

        Parameters
        ----------
        dim
            The embedding dimension.

        See Also
        --------
        petsc.DMSetCoordinateDim

        """
        cdef PetscInt cdim = asInt(dim)
        CHKERR( DMSetCoordinateDim(self.dm, cdim) )

    def setOptionsPrefix(self, prefix: str) -> None:
        """Set the prefix used for searching for options in the database.

        Logically collective.

        See Also
        --------
        petsc_options, getOptionsPrefix, petsc.DMSetOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( DMSetOptionsPrefix(self.dm, cval) )

    def getOptionsPrefix(self) -> str:
        """Return the prefix used for searching for options in the database.

        Not collective.

        See Also
        --------
        petsc_options, setOptionsPrefix, petsc.DMGetOptionsPrefix

        """
        cdef const char *cval = NULL
        CHKERR( DMGetOptionsPrefix(self.dm, &cval) )
        return bytes2str(cval)

    def appendOptionsPrefix(self, prefix: str) -> None:
        """Append to the prefix used for searching for options in the database.

        Logically collective.

        See Also
        --------
        petsc_options, setOptionsPrefix, petsc.DMAppendOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( DMAppendOptionsPrefix(self.dm, cval) )

    def setFromOptions(self) -> None:
        """Configure the object from the options database.

        Collective.

        Notes
        -----
        Options database is available to set up a specific configuration.
        For instance, ``-dm_vec_type`` sets the type of vector to create inside of the `DM`.

        See Also
        --------
        petsc_options, petsc.DMSetFromOptions

        """
        CHKERR( DMSetFromOptions(self.dm) )

    def viewFromOptions(self, name: str, Object obj=None) -> None:
        """View a `DM` based in the options.

        Collective.

        Parameters
        ----------
        name
            Name used to activate the viewing.
        obj
            Object provides the prefix for the options database.

        See Also
        --------
        petsc.DMViewFromOptions

        """
        cdef const char *cname = NULL
        _ = str2bytes(name, &cname)
        cdef PetscObject  cobj = NULL
        if obj is not None: cobj = obj.obj[0]
        CHKERR( DMViewFromOptions(self.dm, cobj, cname) )

    def setUp(self) -> Self:
        """Return the data structure.

        Collective.

        See Also
        --------
        petsc.DMSetUp

        """
        CHKERR( DMSetUp(self.dm) )
        return self

    # --- application context ---

    def setAppCtx(self, appctx):
        self.set_attr('__appctx__', appctx)

    def getAppCtx(self):
        return self.get_attr('__appctx__')

    #

    def setBasicAdjacency(self, useCone: bool, useClosure: bool) -> None:
        """Set the flags for determining variable influence.

        Not collective.

        Parameters
        ----------
        useCone
            If `True`, the variable influence is set, starting with the cone operation.
        useClosure
            If `True`, the variable influence is set using transitive closure.

        See Also
        --------
        petsc.DMSetBasicAdjacency

        """
        cdef PetscBool uC  = useCone
        cdef PetscBool uCl = useClosure
        CHKERR( DMSetBasicAdjacency(self.dm, uC, uCl) )

    def getBasicAdjacency(self) -> tuple[bool, bool]:
        """Return the flags for determining variable influence.

        Not collective.

        Returns
        -------
        toBool(uC) : bool
            This flag provides the variable influence starting with the cone operation if `True`.
        toBool(uCl) : bool
            This flag provides the variable influence using transitive closure.

        See Also
        --------
        petsc.DMGetBasicAdjacency

        """
        cdef PetscBool uC  = PETSC_FALSE
        cdef PetscBool uCl = PETSC_FALSE
        CHKERR( DMGetBasicAdjacency(self.dm, &uC, &uCl) )
        return toBool(uC), toBool(uCl)

    def setFieldAdjacency(self, field: int, useCone: bool, useClosure: bool) -> None:
        """Set the flags for determining variable influence.

        Not collective.

        Parameters
        ----------
        field
            The field number.
        useCone
            If `True`, the variable influence is set, starting with the cone operation.
        useClosure
            If `True`, the variable influence is set using transitive closure.

        See Also
        --------
        petsc.DMSetAdjacency

        """
        cdef PetscInt  f   = asInt(field)
        cdef PetscBool uC  = useCone
        cdef PetscBool uCl = useClosure
        CHKERR( DMSetAdjacency(self.dm, f, uC, uCl) )

    def getFieldAdjacency(self, field: int) -> tuple[bool, bool]:
        """Return the flags for determining variable influence.

        Not collective.

        Parameters
        ----------
        field
            The field number.

        Returns
        -------
        toBool(uC) : bool
            This flag provides the variable influence starting with the cone operation if `True`.
        toBool(uCl) : bool
            This flag provides the variable influence using transitive closure.

        See Also
        --------
        petsc.DMGetAdjacency

        """
        cdef PetscInt  f   = asInt(field)
        cdef PetscBool uC  = PETSC_FALSE
        cdef PetscBool uCl = PETSC_FALSE
        CHKERR( DMGetAdjacency(self.dm, f, &uC, &uCl) )
        return toBool(uC), toBool(uCl)

    #

    def createSubDM(self, fields: Sequence[int]) -> tuple[IS, DM]:
        """Return `IS` and `DM` encapsulating a subproblem.

        Not collective.

        Returns
        -------
        iset : IS
            The global indices for all the degrees of freedom.
        subdm : DM
            The `DM` for the subproblem.

        See Also
        --------
        petsc.DMCreateSubDM

        """
        cdef IS iset = IS()
        cdef DM subdm = DM()
        cdef PetscInt *ifields = NULL
        cdef PetscInt numFields = 0
        fields = iarray_i(fields, &numFields, &ifields)
        CHKERR( DMCreateSubDM( self.dm, numFields, ifields, &iset.iset, &subdm.dm) )
        return iset, subdm

    #

    def setAuxiliaryVec(self, Vec aux, label: DMLabel | None, value=0, part=0) -> None:
        """Set an auxiliary vector for a specific region.

        Not collective.

        Parameters
        ----------
        aux
            The auxiliary vector.
        label
            The name of the `DMLabel`.
        value
            Indicate the region.
        part
            The equation part, or 0 is unused.

        See Also
        --------
        petsc.DMGetLabel, petsc.DMSetAuxiliaryVec

        """
        cdef PetscInt cvalue = asInt(value)
        cdef PetscInt cpart = asInt(part)
        cdef const char *cval = NULL
        cdef PetscDMLabel clbl = NULL
        label = str2bytes(label, &cval)
        if cval == NULL: cval = b"" # XXX Should be fixed upstream
        CHKERR( DMGetLabel(self.dm, cval, &clbl) )
        CHKERR( DMSetAuxiliaryVec(self.dm, clbl, cvalue, cpart, aux.vec) )

    def getAuxiliaryVec(self, label: str | None = None, value: int | None = 0, part: int | None = 0) -> Vec:
        """Return an auxiliary vector for region.

        Not collective.

        Parameters
        ----------
        label
            The name of the `DMLabel`.
        value
            Indicate the region.
        part
            The equation part, or 0 is unused.

        See Also
        --------
        DM.getLabel, petsc.DMGetAuxiliaryVec

        """
        cdef PetscInt cvalue = asInt(value)
        cdef PetscInt cpart = asInt(part)
        cdef const char *cval = NULL
        cdef PetscDMLabel clbl = NULL
        cdef Vec aux = Vec()
        label = str2bytes(label, &cval)
        if cval == NULL: cval = b"" # XXX Should be fixed upstream
        CHKERR( DMGetLabel(self.dm, cval, &clbl) )
        CHKERR( DMGetAuxiliaryVec(self.dm, clbl, cvalue, cpart, &aux.vec) )
        return aux

    def setNumFields(self, numFields: int) -> None:
        """Set the number of fields in the `DM`.

        Logically collective.

        See Also
        --------
        petsc.DMSetNumFields

        """
        cdef PetscInt cnum = asInt(numFields)
        CHKERR( DMSetNumFields(self.dm, cnum) )

    def getNumFields(self) -> int:
        """Return the number of fields in the `DM`.

        Not collective.

        See Also
        --------
        petsc.DMGetNumFields

        """
        cdef PetscInt cnum = 0
        CHKERR( DMGetNumFields(self.dm, &cnum) )
        return toInt(cnum)

    def setField(self, index: int, Object field, label: str | None = None) -> None:
        """Set the discretization object for a given `DM` field.

        Logically collective.

        Parameters
        ----------
        index
            The field number.
        field
            The discretization object.
        label
            The name of the label indicating the support of the field, or `None` for the entire mesh, currently must be `None`.

        See Also
        --------
        petsc.DMSetField

        """
        cdef PetscInt     cidx = asInt(index)
        cdef PetscObject  cobj = field.obj[0]
        cdef PetscDMLabel clbl = NULL
        assert label is None
        CHKERR( DMSetField(self.dm, cidx, clbl, cobj) )

    def getField(self, index: int) -> tuple[Object, None]:
        """Return the discretization object for a given `DM` field.

        Not collective.

        Parameters
        ----------
        index
            The field number.

        See Also
        --------
        petsc.DMGetField

        """
        cdef PetscInt     cidx = asInt(index)
        cdef PetscObject  cobj = NULL
        cdef PetscDMLabel clbl = NULL
        CHKERR( DMGetField(self.dm, cidx, &clbl, &cobj) )
        assert clbl == NULL
        cdef Object field = subtype_Object(cobj)()
        field.obj[0] = cobj
        PetscINCREF(field.obj)
        return (field, None) # TODO REVIEW

    def addField(self, Object field, label: str | None = None) -> None:
        """Add a field to a `DM` object.

        Logically collective.

        Parameters
        ----------
        field
            The discretization object.
        label
            The name of the label indicating the support of the field, or `None` for the entire mesh, currently must be `None`.

        See Also
        --------
        petsc.DMAddField

        """
        cdef PetscObject  cobj = field.obj[0]
        cdef PetscDMLabel clbl = NULL
        assert label is None
        CHKERR( DMAddField(self.dm, clbl, cobj) )

    def clearFields(self) -> None:
        """Remove all fields from the `DM`.

        Logically collective.

        See Also
        --------
        petsc.DMClearFields

        """
        CHKERR( DMClearFields(self.dm) )

    def copyFields(self, DM dm) -> None:
        """Copy the discretizations of this `DM` into another `DM`.

        Collective.

        Parameters
        ----------
        dm
            The `DM` that the fields are copied into.

        See Also
        --------
        petsc.DMCopyFields

        """
        CHKERR( DMCopyFields(self.dm, dm.dm) )

    def createDS(self) -> None:
        """Create discrete systems.

        Collective.

        See Also
        --------
        petsc.DMCreateDS

        """
        CHKERR( DMCreateDS(self.dm) )

    def clearDS(self) -> None:
        """Remove all discrete systems from the `DM`.

        Logically collective.

        See Also
        --------
        petsc.DMClearDS

        """
        CHKERR( DMClearDS(self.dm) )

    def getDS(self) -> DS:
        """Return default `DS`.

        Not collective.

        See Also
        --------
        petsc.DMGetDS

        """
        cdef DS ds = DS()
        CHKERR( DMGetDS(self.dm, &ds.ds) )
        PetscINCREF(ds.obj)
        return ds

    def copyDS(self, DM dm) -> None:
        """Copy the discrete systems for this `DM` into another `DM`.

        Collective.

        Parameters
        ----------
        dm
            The `DM` that the discrete fields are copied into.

        See Also
        --------
        petsc.DMCopyDS

        """
        CHKERR( DMCopyDS(self.dm, dm.dm) )

    def copyDisc(self, DM dm) -> None:
        """Copy fields and discrete systems of a `DM` into another `DM`.

        Collective.

        Parameters
        ----------
        dm
            The `DM` that the fields and discrete systems are copied into.

        See Also
        --------
        petsc.DMCopyDisc

        """
        CHKERR( DMCopyDisc(self.dm, dm.dm) )

    #

    def getBlockSize(self) -> int:
        """Return the inherent block size associated with a `DM`.

        Not collective.

        See Also
        --------
        petsc.DMGetBlockSize

        """
        cdef PetscInt bs = 1
        CHKERR( DMGetBlockSize(self.dm, &bs) )
        return toInt(bs)

    def setVecType(self, vec_type: Vec.Type | str) -> None:
        """Set the type of vector.

        Logically collective.

        See Also
        --------
        Vec.Type, petsc.DMSetVecType

        """
        cdef PetscVecType vtype = NULL
        vec_type = str2bytes(vec_type, &vtype)
        CHKERR( DMSetVecType(self.dm, vtype) )

    def createGlobalVec(self) -> Vec:
        """Return a global vector.

        Collective.

        See Also
        --------
        petsc.DMCreateGlobalVector

        """
        cdef Vec vg = Vec()
        CHKERR( DMCreateGlobalVector(self.dm, &vg.vec) )
        return vg

    def createLocalVec(self) -> Vec:
        """Return a local vector.

        Not collective.

        See Also
        --------
        petsc.DMCreateLocalVector

        """
        cdef Vec vl = Vec()
        CHKERR( DMCreateLocalVector(self.dm, &vl.vec) )
        return vl

    def getGlobalVec(self) -> Vec:
        """Return a global vector.

        Collective on `DM`.

        See Also
        --------
        petsc.DMGetGlobalVector

        """
        cdef Vec vg = Vec()
        CHKERR( DMGetGlobalVector(self.dm, &vg.vec) )
        PetscINCREF(vg.obj)
        return vg

    def restoreGlobalVec(self, Vec vg) -> None:
        """Restore a global vector.

        Not collective.

        Parameters
        ----------
        vg
            The global vector.

        See Also
        --------
        petsc.DMRestoreGlobalVector

        """
        CHKERR( PetscObjectDereference(<PetscObject>vg.vec) )
        CHKERR( DMRestoreGlobalVector(self.dm, &vg.vec) )

    def getLocalVec(self) -> Vec:
        """Return a local vector.

        Not collective.

        See Also
        --------
        petsc.DMGetLocalVector

        """
        cdef Vec vl = Vec()
        CHKERR( DMGetLocalVector(self.dm, &vl.vec) )
        PetscINCREF(vl.obj)
        return vl

    def restoreLocalVec(self, Vec vl) -> None:
        """Restore a local vector.

        Not collective.

        Parameters
        ----------
        vl
            The local vector.

        See Also
        --------
        petsc.DMRestoreLocalVector

        """
        CHKERR( PetscObjectDereference(<PetscObject>vl.vec) )
        CHKERR( DMRestoreLocalVector(self.dm, &vl.vec) )

    def globalToLocal(self, Vec vg, Vec vl, addv: InsertMode | None = None) -> None:
        """Update local vectors from global vector.

        Neighborwise collective.

        Parameters
        ----------
        vg
            The global vector.
        vl
            The local vector.
        addv
            `InsertMode.INSERT_VALUES` or `InsertMode.ADD_VALUES`.

        See Also
        --------
        petsc.DMGlobalToLocalBegin, petsc.DMGlobalToLocalEnd

        """
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMGlobalToLocalBegin(self.dm, vg.vec, im, vl.vec) )
        CHKERR( DMGlobalToLocalEnd  (self.dm, vg.vec, im, vl.vec) )

    def localToGlobal(self, Vec vl, Vec vg, addv: InsertMode | None = None) -> None:
        """Update global vectors from local vector.

        Neighborwise collective.

        Parameters
        ----------
        vg
            The global vector.
        vl
            The local vector.
        addv
            If `InsertMode.INSERT_VALUES`, then no parallel communication is used, while, if `InsertMode.ADD_VALUES`, then all ghost points from the same base point accumulate into that base point.

        See Also
        --------
        petsc.DMLocalToGlobalBegin, petsc.DMLocalToGlobalEnd

        """
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMLocalToGlobalBegin(self.dm, vl.vec, im, vg.vec) )
        CHKERR( DMLocalToGlobalEnd(self.dm, vl.vec, im, vg.vec) )

    def localToLocal(self, Vec vl, Vec vlg, addv: InsertMode | None = None) -> None:
        """Map the values from a local vector to another local vector.

        Neighborwise collective.

        Parameters
        ----------
        vg
            The global vector.
        vl
            The local vector.
        addv
            `InsertMode.INSERT_VALUES` or `InsertMode.ADD_VALUES`.

        See Also
        --------
        petsc.DMLocalToLocalBegin, petsc.DMLocalToLocalEnd

        """
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMLocalToLocalBegin(self.dm, vl.vec, im, vlg.vec) )
        CHKERR( DMLocalToLocalEnd  (self.dm, vl.vec, im, vlg.vec) )

    def getLGMap(self) -> LGMap:
        """Return local mapping to global mapping.

        Collective.

        See Also
        --------
        petsc.DMGetLocalToGlobalMapping

        """
        cdef LGMap lgm = LGMap()
        CHKERR( DMGetLocalToGlobalMapping(self.dm, &lgm.lgm) )
        PetscINCREF(lgm.obj)
        return lgm

    #

    def getCoordinateDM(self) -> DM:
        """Return the coordinate `DM`.

        Collective.

        See Also
        --------
        petsc.DMGetCoordinateDM

        """
        cdef DM cdm = type(self)()
        CHKERR( DMGetCoordinateDM(self.dm, &cdm.dm) )
        PetscINCREF(cdm.obj)
        return cdm

    def getCoordinateSection(self) -> Section:
        """Return coordinate values layout over the mesh.

        Collective.

        See Also
        --------
        petsc.DMGetCoordinateSection

        """
        cdef Section sec = Section()
        CHKERR( DMGetCoordinateSection(self.dm, &sec.sec) )
        PetscINCREF(sec.obj)
        return sec

    def setCoordinates(self, Vec c) -> None:
        """Set a global vector that holds the coordinates.

        Collective.

        Parameters
        ----------
        c
            Coordinate Vector.

        See Also
        --------
        petsc.DMSetCoordinates

        """
        CHKERR( DMSetCoordinates(self.dm, c.vec) )

    def getCoordinates(self) -> Vec:
        """Return a global vector with the coordinates associated.

        Collective.

        See Also
        --------
        petsc.DMGetCoordinates

        """
        cdef Vec c = Vec()
        CHKERR( DMGetCoordinates(self.dm, &c.vec) )
        PetscINCREF(c.obj)
        return c

    def setCoordinatesLocal(self, Vec c) -> None:
        """Set a local vector with the ghost point holding the coordinates.

        Not collective.

        Parameters
        ----------
        c
            Coordinate Vector.

        See Also
        --------
        petsc.DMSetCoordinatesLocal

        """
        CHKERR( DMSetCoordinatesLocal(self.dm, c.vec) )

    def getCoordinatesLocal(self) -> Vec:
        """Return a local vector with the coordinates associated.

        Collective the first time it is called.

        See Also
        --------
        petsc.DMGetCoordinatesLocal

        """
        cdef Vec c = Vec()
        CHKERR( DMGetCoordinatesLocal(self.dm, &c.vec) )
        PetscINCREF(c.obj)
        return c

    def projectCoordinates(self, FE disc) -> Self:
        """Project coordinates to a different space.

        Parameters
        ----------
        disc
            The new coordinates discretization.

        See Also
        --------
        petsc.DMProjectCoordinates

        """
        CHKERR( DMProjectCoordinates(self.dm, disc.fe))
        return self

    def getBoundingBox(self) -> tuple[tuple[float, float], ...]:
        """Return the dimension of embedding space for coordinates values.

        Not collective.

        See Also
        --------
        petsc.DMGetBoundingBox

        """
        cdef PetscInt i,dim=0
        CHKERR( DMGetCoordinateDim(self.dm, &dim) )
        cdef PetscReal gmin[3], gmax[3]
        CHKERR( DMGetBoundingBox(self.dm, gmin, gmax) )
        return tuple([(toReal(gmin[i]), toReal(gmax[i]))
                      for i from 0 <= i < dim])

    def getLocalBoundingBox(self) -> tuple[tuple[float, float], ...]:
        """Return the bounding box for the piece of the `DM`.

        Not collective.

        See Also
        --------
        petsc.DMGetLocalBoundingBox

        """
        cdef PetscInt i,dim=0
        CHKERR( DMGetCoordinateDim(self.dm, &dim) )
        cdef PetscReal lmin[3], lmax[3]
        CHKERR( DMGetLocalBoundingBox(self.dm, lmin, lmax) )
        return tuple([(toReal(lmin[i]), toReal(lmax[i]))
                      for i from 0 <= i < dim])

    def localizeCoordinates(self) -> None:
        """Create local coordinates for cells having periodic faces.

        Collective.

        Notes
        -----
        Used if the mesh is periodic.

        See Also
        --------
        petsc.DMLocalizeCoordinates

        """
        CHKERR( DMLocalizeCoordinates(self.dm) )
    #

    def setMatType(self, mat_type: Mat.Type | str) -> None:
        """Set matrix type to be used by `DM.createMat`.

        Logically collective.

        Parameters
        ----------
        mat_type
            The matrix type.

        Notes
        -----
        The option ``-dm_mat_type`` is used to set the matrix type.

        See Also
        --------
        petsc_options, petsc.DMSetMatType

        """
        cdef PetscMatType mtype = NULL
        mat_type = str2bytes(mat_type, &mtype)
        CHKERR( DMSetMatType(self.dm, mtype) )

    def createMat(self) -> Mat:
        """Return an empty matrix.

        Collective.

        See Also
        --------
        petsc.DMCreateMatrix

        """
        cdef Mat mat = Mat()
        CHKERR( DMCreateMatrix(self.dm, &mat.mat) )
        return mat

    def createMassMatrix(self, DM dmf) -> Mat:
        """Return the mass matrix between this `DM` and the given `DM`.

        Collective.

        Parameters
        ----------
        dmf
            The second `DM`.

        See Also
        --------
        petsc.DMCreateMassMatrix

        """
        cdef Mat mat = Mat()
        CHKERR( DMCreateMassMatrix(self.dm, dmf.dm, &mat.mat) )
        return mat

    def createInterpolation(self, DM dm) -> tuple[Mat, Vec]:
        """Return the interpolation matrix to a finer `DM`.

        Collective.

        Parameters
        ----------
        dm
            The second, finer `DM`.

        See Also
        --------
        petsc.DMCreateInterpolation

        """
        cdef Mat A = Mat()
        cdef Vec scale = Vec()
        CHKERR( DMCreateInterpolation(self.dm, dm.dm,
                                   &A.mat, &scale.vec))
        return (A, scale)

    def createInjection(self, DM dm) -> Mat:
        """Return the injection matrix into a finer `DM`.

        Collective.

        Parameters
        ----------
        dm
            The second, finer `DM` object.

        See Also
        --------
        petsc.DMCreateInjection

        """
        cdef Mat inject = Mat()
        CHKERR( DMCreateInjection(self.dm, dm.dm, &inject.mat) )
        return inject

    def createRestriction(self, DM dm) -> Mat:
        """Return the restriction matrix between this `DM` and the given `DM`.

        Collective.

        Parameters
        ----------
        dm
            The second, finer `DM` object.

        See Also
        --------
        petsc.DMCreateRestriction

        """
        cdef Mat mat = Mat()
        CHKERR( DMCreateRestriction(self.dm, dm.dm, &mat.mat) )
        return mat

    def convert(self, dm_type: DM.Type | str) -> DM:
        """Return a `DM` converted to another `DM`.

        Collective.

        Parameters
        ----------
        dm_type
            The new `DM.Type`, use ``“same”`` for the same type.

        See Also
        --------
        DM.Type, petsc.DMConvert

        """
        cdef PetscDMType cval = NULL
        dm_type = str2bytes(dm_type, &cval)
        cdef PetscDM newdm = NULL
        CHKERR( DMConvert(self.dm, cval, &newdm) )
        cdef DM dm = <DM>subtype_DM(newdm)()
        dm.dm = newdm
        return dm

    def refine(self, comm: Comm | None = None) -> DM:
        """Return a refined `DM` object.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.DMRefine

        """
        cdef MPI_Comm dmcomm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.dm, &dmcomm) )
        dmcomm = def_Comm(comm, dmcomm)
        cdef PetscDM newdm = NULL
        CHKERR( DMRefine(self.dm, dmcomm, &newdm) )
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        return dm

    def coarsen(self, comm: Comm | None = None) -> DM:
        """Return a coarsened `DM` object.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.DMCoarsen

        """
        cdef MPI_Comm dmcomm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.dm, &dmcomm) )
        dmcomm = def_Comm(comm, dmcomm)
        cdef PetscDM newdm = NULL
        CHKERR( DMCoarsen(self.dm, dmcomm, &newdm) )
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        return dm

    def refineHierarchy(self, nlevels: int) -> list:
        """Refine this `DM` and return the refined `DM` hierarchy.

        Collective.

        Parameters
        ----------
        nlevels
            The number of levels of refinement.

        See Also
        --------
        petsc.DMRefineHierarchy

        """
        cdef PetscInt i, n = asInt(nlevels)
        cdef PetscDM *newdmf = NULL
        cdef object tmp = oarray_p(empty_p(n), NULL, <void**>&newdmf)
        CHKERR( DMRefineHierarchy(self.dm, n, newdmf) )
        cdef DM dmf = None
        cdef list hierarchy = []
        for i from 0 <= i < n:
            dmf = subtype_DM(newdmf[i])()
            dmf.dm = newdmf[i]
            hierarchy.append(dmf)
        return hierarchy

    def coarsenHierarchy(self, nlevels: int) -> list:
        """Coarsen this `DM` and return the coarsened `DM` hierarchy.

        Collective.

        Parameters
        ----------
        nlevels
            The number of levels of coarsening.

        See Also
        --------
        petsc.DMCoarsenHierarchy

        """
        cdef PetscInt i, n = asInt(nlevels)
        cdef PetscDM *newdmc = NULL
        cdef object tmp = oarray_p(empty_p(n),NULL, <void**>&newdmc)
        CHKERR( DMCoarsenHierarchy(self.dm, n, newdmc) )
        cdef DM dmc = None
        cdef list hierarchy = []
        for i from 0 <= i < n:
            dmc = subtype_DM(newdmc[i])()
            dmc.dm = newdmc[i]
            hierarchy.append(dmc)
        return hierarchy

    def getRefineLevel(self) -> int:
        """Return the refinement level.

        Not collective.

        See Also
        --------
        petsc.DMGetRefineLevel

        """
        cdef PetscInt n = 0
        CHKERR( DMGetRefineLevel(self.dm, &n) )
        return toInt(n)

    def setRefineLevel(self, level: int) -> None:
        """Set the number of refinements.

        Not collective.

        Parameters
        ----------
        nlevels
            The number of refinement.

        See Also
        --------
        petsc.DMSetRefineLevel

        """
        cdef PetscInt clevel = asInt(level)
        CHKERR( DMSetRefineLevel(self.dm, clevel) )

    def getCoarsenLevel(self) -> int:
        """Return the number of coarsenings.

        Not collective.

        See Also
        --------
        petsc.DMGetCoarsenLevel

        """
        cdef PetscInt n = 0
        CHKERR( DMGetCoarsenLevel(self.dm, &n) )
        return toInt(n)

    #

    def adaptLabel(self, label: str) -> DM:
        """Adapt a `DM` based on a `DMLabel`.

        Collective.

        Parameters
        ----------
        label
            The name of the `DMLabel`.

        See Also
        --------
        petsc.DMAdaptLabel

        """
        cdef const char *cval = NULL
        cdef PetscDMLabel clbl = NULL
        label = str2bytes(label, &cval)
        CHKERR( DMGetLabel(self.dm, cval, &clbl) )
        cdef DM newdm = DMPlex()
        CHKERR( DMAdaptLabel(self.dm, clbl, &newdm.dm) )
        return newdm

    def adaptMetric(
        self,
        Vec metric,
        bdLabel: str | None = None,
        rgLabel: str | None = None,
    ) -> DM:
        """Return a mesh adapted to the specified metric field.

        Parameters
        ----------
        metric
            The metric to which the mesh is adapted, defined vertex-wise.
        bdLabel
            Label for boundary tags.
        rgLabel
            Label for cell tag.

        See Also
        --------
        petsc.DMAdaptMetric

        """
        cdef const char *cval = NULL
        cdef PetscDMLabel cbdlbl = NULL
        cdef PetscDMLabel crglbl = NULL
        bdLabel = str2bytes(bdLabel, &cval)
        if cval == NULL: cval = b"" # XXX Should be fixed upstream
        CHKERR( DMGetLabel(self.dm, cval, &cbdlbl) )
        rgLabel = str2bytes(rgLabel, &cval)
        if cval == NULL: cval = b"" # XXX Should be fixed upstream
        CHKERR( DMGetLabel(self.dm, cval, &crglbl) )
        cdef DM newdm = DMPlex()
        CHKERR( DMAdaptMetric(self.dm, metric.vec, cbdlbl, crglbl, &newdm.dm) )
        return newdm

    def getLabel(self, name: str) -> DMLabel:
        """Return the label of a given name.

        Not collective.

        See Also
        --------
        petsc.DMGetLabel

        """
        cdef const char *cname = NULL
        cdef DMLabel dmlabel = DMLabel()
        name = str2bytes(name, &cname)
        CHKERR( DMGetLabel(self.dm, cname, &dmlabel.dmlabel) )
        PetscINCREF(dmlabel.obj)
        return dmlabel

    #

    def setLocalSection(self, Section sec) -> None:
        """Set the `Section` encoding the local data layout for the `DM`.

        See Also
        --------
        petsc.DMSetLocalSection

        """
        CHKERR( DMSetLocalSection(self.dm, sec.sec) )

    def getLocalSection(self) -> Section:
        """Return the `Section` encoding the local data layout for the `DM`.

        See Also
        --------
        petsc.DMGetGlobalSection

        """
        cdef Section sec = Section()
        CHKERR( DMGetLocalSection(self.dm, &sec.sec) )
        PetscINCREF(sec.obj)
        return sec

    def setGlobalSection(self, Section sec) -> None:
        """Set the `Section` encoding the global data layout for the `DM`.

        See Also
        --------
        petsc.DMSetGlobalSection

        """
        CHKERR( DMSetGlobalSection(self.dm, sec.sec) )

    def getGlobalSection(self) -> Section:
        """Return the `Section` encoding the global data layout for the `DM`.

        See Also
        --------
        petsc.DMGetGlobalSection

        """
        cdef Section sec = Section()
        CHKERR( DMGetGlobalSection(self.dm, &sec.sec) )
        PetscINCREF(sec.obj)
        return sec

    setSection = setLocalSection
    getSection = getLocalSection
    setDefaultSection = setLocalSection
    getDefaultSection = getLocalSection
    setDefaultLocalSection = setLocalSection
    getDefaultLocalSection = getLocalSection
    setDefaultGlobalSection = setGlobalSection
    getDefaultGlobalSection = getGlobalSection

    def createSectionSF(self, Section localsec, Section globalsec) -> None:
        """Create the `SF` encoding the parallel DOF overlap for the `DM`.

        Parameters
        ----------
        localsec
            Describe the local data layout.
        globalsec
            Describe the global data layout.

        Note
        ----
        Encoding based on the `Section` describing the data layout.

        See Also
        --------
        DM.getSectionSF, petsc.DMCreateSectionSF

        """
        CHKERR( DMCreateSectionSF(self.dm, localsec.sec, globalsec.sec) )

    def getSectionSF(self) -> SF:
        """Return the `Section` encoding the parallel DOF overlap.

        See Also
        --------
        petsc.DMGetSectionSF

        """
        cdef SF sf = SF()
        CHKERR( DMGetSectionSF(self.dm, &sf.sf) )
        PetscINCREF(sf.obj)
        return sf

    def setSectionSF(self, SF sf) -> None:
        """Set the `Section` encoding the parallel DOF overlap for the `DM`.

        See Also
        --------
        petsc.DMSetSectionSF

        """
        CHKERR( DMSetSectionSF(self.dm, sf.sf) )

    createDefaultSF = createSectionSF
    getDefaultSF = getSectionSF
    setDefaultSF = setSectionSF

    def getPointSF(self) -> SF:
        """Return the `SF` encoding the parallel DOF overlap for the `DM`.

        See Also
        --------
        petsc.DMGetPointSF

        """
        cdef SF sf = SF()
        CHKERR( DMGetPointSF(self.dm, &sf.sf) )
        PetscINCREF(sf.obj)
        return sf

    def setPointSF(self, SF sf) -> None:
        """Set the `SF` encoding the parallel DOF overlap for the `DM`.

        See Also
        --------
        petsc.DMSetPointSF

        """
        CHKERR( DMSetPointSF(self.dm, sf.sf) )

    def getNumLabels(self) -> int:
        """Return the number of labels defined by on the `DM`.

        Not collective.

        See Also
        --------
        petsc.DMGetNumLabels

        """
        cdef PetscInt nLabels = 0
        CHKERR( DMGetNumLabels(self.dm, &nLabels) )
        return toInt(nLabels)

    def getLabelName(self, index: int) -> str:
        """Return the name of nth label.

        Not collective.

        Parameters
        ----------
        index
            The label number.

        See Also
        --------
        petsc.DMGetLabelName

        """
        cdef PetscInt cindex = asInt(index)
        cdef const char *cname = NULL
        CHKERR( DMGetLabelName(self.dm, cindex, &cname) )
        return bytes2str(cname)

    def hasLabel(self, name: str) -> bool:
        """Determine whether the `DM` has a label.

        Not collective.

        Parameters
        ----------
        name
            The label name.

        See Also
        --------
        petsc.DMHasLabel

        """
        cdef PetscBool flag = PETSC_FALSE
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMHasLabel(self.dm, cname, &flag) )
        return toBool(flag)

    def createLabel(self, name: str) -> None:
        """Create a label of the given name if it does not already exist.

        Not collective.

        Parameters
        ----------
        name
            The label name.

        See Also
        --------
        petsc.DMCreateLabel

        """
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMCreateLabel(self.dm, cname) )

    def removeLabel(self, name: str) -> None:
        """Remove and destroy the label by name.

        Not collective.

        Parameters
        ----------
        name
            The label name.

        See Also
        --------
        petsc.DMRemoveLabel

        """
        cdef const char *cname = NULL
        cdef PetscDMLabel clbl = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMRemoveLabel(self.dm, cname, &clbl) )
        # TODO: Once DMLabel is wrapped, this should return the label, like the C function.
        CHKERR( DMLabelDestroy(&clbl) )

    def getLabelValue(self, name: str, point: int) -> int:
        """Return the value in `DMLabel` for the given point.

        Not collective.

        Parameters
        ----------
        name
            The label name.
        point
            The mesh point

        See Also
        --------
        petsc.DMGetLabelValue

        """
        cdef PetscInt cpoint = asInt(point), value = 0
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMGetLabelValue(self.dm, cname, cpoint, &value) )
        return toInt(value)

    def setLabelValue(self, name: str, point: int, value: int) -> None:
        """Set a point to a `DMLabel` with a given value.

        Not collective.

        Parameters
        ----------
        name
            The label name.
        point
            The mesh point.
        value
            The label value for the point.

        See Also
        --------
        petsc.DMSetLabelValue

        """
        cdef PetscInt cpoint = asInt(point), cvalue = asInt(value)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMSetLabelValue(self.dm, cname, cpoint, cvalue) )

    def clearLabelValue(self, name: str, point: int, value: int) -> None:
        """Remove a point from a `DMLabel` with given value.

        Not collective.

        Parameters
        ----------
        name
            The label name.
        point
            The mesh point.
        value
            The label value for the point.

        See Also
        --------
        petsc.DMClearLabelValue

        """
        cdef PetscInt cpoint = asInt(point), cvalue = asInt(value)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMClearLabelValue(self.dm, cname, cpoint, cvalue) )

    def getLabelSize(self, name: str) -> int:
        """Return the number of values that the `DMLabel` takes.

        Not collective.

        Parameters
        ----------
        name
            The label name.

        See Also
        --------
        petsc.DMLabelGetNumValues, petsc.DMGetLabelSize

        """
        cdef PetscInt size = 0
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMGetLabelSize(self.dm, cname, &size) )
        return toInt(size)

    def getLabelIdIS(self, name: str) -> IS:
        """Return an `IS` of all values that the `DMLabel` takes.

        Not collective.

        Parameters
        ----------
        name
            The label name.

        See Also
        --------
        petsc.DMLabelGetValueIS, petsc.DMGetLabelIdIS

        """
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef IS lis = IS()
        CHKERR( DMGetLabelIdIS(self.dm, cname, &lis.iset) )
        return lis

    def getStratumSize(self, name: str, value: int) -> int:
        """Return the number of points in a label stratum.

        Not collective.

        Parameters
        ----------
        name
            The label name.
        value
            The stratum value.

        See Also
        --------
        petsc.DMGetStratumSize

        """
        cdef PetscInt size = 0
        cdef PetscInt cvalue = asInt(value)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMGetStratumSize(self.dm, cname, cvalue, &size) )
        return toInt(size)

    def getStratumIS(self, name: str, value: int) -> IS:
        """Return the points in a label stratum.

        Not collective.

        Parameters
        ----------
        name
            The label name.
        value
            The stratum value.

        See Also
        --------
        petsc.DMGetStratumIS

        """
        cdef PetscInt cvalue = asInt(value)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef IS sis = IS()
        CHKERR( DMGetStratumIS(self.dm, cname, cvalue, &sis.iset) )
        return sis

    def clearLabelStratum(self, name: str, value: int) -> None:
        """Remove all points from a stratum.

        Not collective.

        Parameters
        ----------
        name
            The label name.
        value
            The stratum value.

        See Also
        --------
        petsc.DMClearLabelStratum

        """
        cdef PetscInt cvalue = asInt(value)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMClearLabelStratum(self.dm, cname, cvalue) )

    def setLabelOutput(self, name: str, output: bool) -> None:
        """Set if a given lable should be saved to a view.

        Not collective.

        Parameters
        ----------
        name
            The label name.
        output
            If `True`, the label is saved to the viewer.

        See Also
        --------
        petsc.DMSetLabelOutput

        """
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscBool coutput = output
        CHKERR( DMSetLabelOutput(self.dm, cname, coutput) )

    def getLabelOutput(self, name: str) -> bool:
        """Return the output flag for a given label.

        Not collective.

        Parameters
        ----------
        name
            The label name.

        See Also
        --------
        petsc.DMGetLabelOutput

        """
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscBool coutput = PETSC_FALSE
        CHKERR( DMGetLabelOutput(self.dm, cname, &coutput) )
        return coutput

    # backward compatibility
    createGlobalVector = createGlobalVec
    createLocalVector = createLocalVec
    getMatrix = createMatrix = createMat

    def setKSPComputeOperators(
        self, operators,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None,
        ) -> None:
        """Matrix associated with the linear system.

        Collective.

        Parameters
        ----------
        operator
            Callback function to compute the operators.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        See Also
        --------
        petsc.DMKSPSetComputeOperators

        """
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (operators, args, kargs)
        self.set_attr('__operators__', context)
        CHKERR( DMKSPSetComputeOperators(self.dm, KSP_ComputeOps, <void*>context) )

    def createFieldDecomposition(self) -> tuple[list, list, list] :
        """Return a list of `IS` objects.

        Not collective.

        Notes
        -----
        The user is responsible for freeing all requested arrays.

        See Also
        --------
        petsc.DMCreateFieldDecomposition

        """
        cdef PetscInt clen = 0
        cdef PetscIS *cis = NULL
        cdef PetscDM *cdm = NULL
        cdef char** cnamelist = NULL

        CHKERR( DMCreateFieldDecomposition(self.dm, &clen, &cnamelist, &cis, &cdm) )

        cdef list isets = [ref_IS(cis[i]) for i from 0 <= i < clen]
        cdef list dms   = []
        cdef list names = []
        cdef DM dm = None

        for i from 0 <= i < clen:
            if cdm != NULL:
                dm = subtype_DM(cdm[i])()
                dm.dm = cdm[i]
                PetscINCREF(dm.obj)
                dms.append(dm)
            else:
                dms.append(None)

            name = bytes2str(cnamelist[i])
            names.append(name)
            CHKERR( PetscFree(cnamelist[i]) )

            CHKERR( ISDestroy(&cis[i]) )
            CHKERR( DMDestroy(&cdm[i]) )

        CHKERR( PetscFree(cis) )
        CHKERR( PetscFree(cdm) )
        CHKERR( PetscFree(cnamelist) )

        return (names, isets, dms) # TODO REVIEW

    def setSNESFunction(
        self, function: SNESFunction,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None,
        ) -> None:

        """Set `SNES` residual evaluation function.

        Not collective.

        Parameters
        ----------
        function
            The callback.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        See Also
        --------
        SNES.setFunction, petsc.DMSNESSetFunction

        """
        if function is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (function, args, kargs)
            self.set_attr('__function__', context)
            CHKERR( DMSNESSetFunction(self.dm, SNES_Function, <void*>context) )
        else:
            CHKERR( DMSNESSetFunction(self.dm, NULL, NULL) )

    def setSNESJacobian(
            self, jacobian: SNESJacobianFunction,
            args: tuple[Any, ...] | None = None,
            kargs: dict[str, Any] | None = None,
            ) -> None:
        """Set the `SNES` Jacobian evaluation function.

        Not collective

        Parameters
        ----------
        jacobian
            The Jacobian callback.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        See Also
        --------
        SNES.setJacobian, petsc.DMSNESSetJacobian

        """
        if jacobian is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (jacobian, args, kargs)
            self.set_attr('__jacobian__', context)
            CHKERR( DMSNESSetJacobian(self.dm, SNES_Jacobian, <void*>context) )
        else:
            CHKERR( DMSNESSetJacobian(self.dm, NULL, NULL) )

    def addCoarsenHook(
        self,
        coarsenhook: DMCoarsenHookFunction,
        restricthook: DMRestrictHookFunction,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None,
    ) -> None:
        """Add a callback to be executed when restricting to a coarser grid.

        Logically collective.

        Parameters
        ----------
        coarsenhook
            The coarsen hook function.
        restricthook
            The restrict hook function.
        args
            Positional arguments for the hooks.
        kargs
            Keyword arguments for the hooks.

        See Also
        --------
        petsc.DMCoarsenHookAdd

        """
        if args  is None: args  = ()
        if kargs is None: kargs = {}

        if coarsenhook is not None:
            coarsencontext = (coarsenhook, args, kargs)

            coarsenhooks = self.get_attr('__coarsenhooks__')
            if coarsenhooks is None:
                coarsenhooks = [coarsencontext]
                CHKERR( DMCoarsenHookAdd(self.dm, DM_PyCoarsenHook, NULL, <void*>NULL) )
            else:
                coarsenhooks.append(coarsencontext)
            self.set_attr('__coarsenhooks__', coarsenhooks)

        if restricthook is not None:
            restrictcontext = (restricthook, args, kargs)

            restricthooks = self.get_attr('__restricthooks__')
            if restricthooks is None:
                restricthooks = [restrictcontext]
                CHKERR( DMCoarsenHookAdd(self.dm, NULL, DM_PyRestrictHook, <void*>NULL) )
            else:
                restricthooks.append(restrictcontext)
            self.set_attr('__restricthooks__', restricthooks)

    # --- application context ---

    property appctx:
        def __get__(self):
            return self.getAppCtx()
        def __set__(self, value):
            self.setAppCtx(value)

    # --- discretization space ---

    property ds:
        def __get__(self):
            return self.getDS()
        def __set__(self, value):
            self.setDS(value)

# --------------------------------------------------------------------

del DMType
del DMBoundaryType
del DMPolytopeType

# --------------------------------------------------------------------
