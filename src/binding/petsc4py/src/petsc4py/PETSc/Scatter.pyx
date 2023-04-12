# --------------------------------------------------------------------

class ScatterType(object):
    """Scatter type.

    See Also
    --------
    petsc.VecScatterType

    """
    BASIC      = S_(PETSCSFBASIC)
    NEIGHBOR   = S_(PETSCSFNEIGHBOR)
    ALLGATHERV = S_(PETSCSFALLGATHERV)
    ALLGATHER  = S_(PETSCSFALLGATHER)
    GATHERV    = S_(PETSCSFGATHERV)
    GATHER     = S_(PETSCSFGATHER)
    ALLTOALL   = S_(PETSCSFALLTOALL)
    WINDOW     = S_(PETSCSFWINDOW)

# --------------------------------------------------------------------

cdef class Scatter(Object):
    """Scatter object.

    The object used to perform data movement between vectors.
    Scatter is described in the `PETSc manual <petsc:sec_scatter>`.

    See Also
    --------
    Vec, SF, petsc.VecScatter

    """


    Type = ScatterType
    Mode = ScatterMode

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.sct
        self.sct = NULL

    def __call__(self, x, y, addv=None, mode=None):
        """Perform the scatter.

        Collective.

        See Also
        --------
        scatter

        """
        self.scatter(x, y, addv, mode)

    #

    def view(self, Viewer viewer=None) -> None:
        """View the scatter.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        petsc.VecScatterView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( VecScatterView(self.sct, vwr) )

    def destroy(self) -> Self:
        """Destroy the scatter.

        Collective.

        See Also
        --------
        petsc.VecScatterDestroy

        """
        CHKERR( VecScatterDestroy(&self.sct) )
        return self

    def create(
        self,
        Vec vec_from,
        IS is_from: IS | None,
        Vec vec_to,
        IS is_to: IS | None,
    ) -> Self:
        """Create a scatter object.

        Collective.

        Parameters
        ----------
        vec_from
            Representative vector from which to scatter the data.
        is_from
            Indices of ``vec_from`` to scatter. If `None`, use all indices.
        vec_to
            Representative vector to which scatter the data.
        is_to
            Indices of ``vec_to`` where to receive. If `None`, use all indices.

        Examples
        --------
        The scatter object can be used to repeatedly perform data movement.
        It is the PETSc equivalent of NumPy-like indexing and slicing,
        with support for parallel communications:

        >>> revmode = PETSc.Scatter.Mode.REVERSE
        >>> v1 = PETSc.Vec().createWithArray([1, 2, 3])
        >>> v2 = PETSc.Vec().createWithArray([0, 0, 0])
        >>> sct = PETSc.Scatter().create(v1,None,v2,None)
        >>> sct.scatter(v1,v2) # v2[:] = v1[:]
        >>> sct.scatter(v2,v1,mode=revmode) # v1[:] = v2[:]

        >>> revmode = PETSc.Scatter.Mode.REVERSE
        >>> v1 = PETSc.Vec().createWithArray([1, 2, 3, 4])
        >>> v2 = PETSc.Vec().createWithArray([0, 0])
        >>> is1 = PETSc.IS().createStride(2, 3, -2)
        >>> sct = PETSc.Scatter().create(v1,is1,v2,None)
        >>> sct.scatter(v1,v2) # v2[:] = v1[3:0:-2]
        >>> sct.scatter(v2,v1,mode=revmode) # v1[3:0:-2] = v2[:]

        See Also
        --------
        IS, petsc.VecScatterCreate

        """
        cdef PetscIS cisfrom = NULL, cisto = NULL
        if is_from is not None: cisfrom = is_from.iset
        if is_to   is not None: cisto   = is_to.iset
        cdef PetscScatter newsct = NULL
        CHKERR( VecScatterCreate(
                vec_from.vec, cisfrom, vec_to.vec, cisto, &newsct) )
        PetscCLEAR(self.obj); self.sct = newsct
        return self

    def setType(self, scatter_type: Type | str) -> None:
        """Set the type of the scatter.

        Logically collective.

        See Also
        --------
        getType, petsc.VecScatterSetType

        """
        cdef PetscScatterType cval = NULL
        vec_type = str2bytes(scatter_type, &cval)
        CHKERR( VecScatterSetType(self.sct, cval) )

    def getType(self) -> str:
        """Return the type of the scatter.

        Not collective.

        See Also
        --------
        setType, petsc.VecScatterGetType

        """
        cdef PetscScatterType cval = NULL
        CHKERR( VecScatterGetType(self.sct, &cval) )
        return bytes2str(cval)

    def setFromOptions(self) -> None:
        """Configure the scatter from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.VecScatterSetFromOptions

        """
        CHKERR( VecScatterSetFromOptions(self.sct) )

    def setUp(self) -> Self:
        """Set up the internal data structures for using the scatter.

        Collective.

        See Also
        --------
        petsc.VecScatterSetUp

        """
        CHKERR( VecScatterSetUp(self.sct) )
        return self

    def copy(self) -> Scatter:
        """Return a copy of the scatter."""
        cdef Scatter scatter = Scatter()
        CHKERR( VecScatterCopy(self.sct, &scatter.sct) )
        return scatter

    @classmethod
    def toAll(cls, Vec vec) -> tuple[Scatter, Vec]:
        """Create a scatter that communicates a vector to all sharing processes.

        Collective.

        Parameters
        ----------
        vec
            The vector to scatter from.

        Notes
        -----
        The created scatter will have the same communicator of ``vec``.
        The method also returns an output vector of appropriate size to
        contain the result of the operation.

        See Also
        --------
        toZero, petsc.VecScatterCreateToAll

        """
        cdef Scatter scatter = Scatter()
        cdef Vec ovec = Vec()
        CHKERR( VecScatterCreateToAll(
            vec.vec, &scatter.sct, &ovec.vec) )
        return (scatter, ovec)

    @classmethod
    def toZero(cls, Vec vec) -> tuple[Scatter, Vec]:
        """Create a scatter that communicates a vector to rank zero.

        Collective.

        Parameters
        ----------
        vec
            The vector to scatter from.

        Notes
        -----
        The created scatter will have the same communicator of ``vec``.
        The method also returns an output vector of appropriate size to
        contain the result of the operation.

        See Also
        --------
        toAll, petsc.VecScatterCreateToZero

        """
        cdef Scatter scatter = Scatter()
        cdef Vec ovec = Vec()
        CHKERR( VecScatterCreateToZero(
            vec.vec, &scatter.sct, &ovec.vec) )
        return (scatter, ovec)
    #

    def begin(
        self,
        Vec vec_from,
        Vec vec_to,
        addv: InsertModeSpec = None,
        mode: ScatterModeSpec = None,
    ) -> None:
        """Begin a generalized scatter from one vector into another.

        Collective.

        This call has to be concluded with a call to `end`.
        For additional details on the Parameters, see `scatter`.

        See Also
        --------
        create, end, petsc.VecScatterBegin

        """
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR( VecScatterBegin(self.sct, vec_from.vec, vec_to.vec,
                                caddv, csctm) )

    def end(
        self,
        Vec vec_from,
        Vec vec_to,
        addv: InsertModeSpec = None,
        mode: ScatterModeSpec = None,
    ) -> None:
        """Complete a generalized scatter from one vector into another.

        Collective.

        This call has to be preceded by a call to `begin`.
        For additional details on the Parameters, see `scatter`.

        See Also
        --------
        create, begin, petsc.VecScatterEnd

        """
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR( VecScatterEnd(self.sct, vec_from.vec, vec_to.vec,
                              caddv, csctm) )

    def scatter(
        self,
        Vec vec_from,
        Vec vec_to,
        addv: InsertModeSpec = None,
        mode: ScatterModeSpec = None,
    ) -> None:
        """Perform a generalized scatter from one vector into another.

        Collective.

        Parameters
        ----------
        vec_from
            The source vector.
        vec_to
            The destination vector.
        addv
            Insertion mode.
        mode
            Scatter mode.

        See Also
        --------
        create, begin, end, petsc.VecScatterBegin, petsc.VecScatterEnd

        """
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR( VecScatterBegin(self.sct, vec_from.vec, vec_to.vec,
                                caddv, csctm) )
        CHKERR( VecScatterEnd(self.sct, vec_from.vec, vec_to.vec,
                              caddv, csctm) )

    scatterBegin = begin
    scatterEnd = end

# --------------------------------------------------------------------

del ScatterType

# --------------------------------------------------------------------
