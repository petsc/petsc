
cdef class DMInterpolation:
    """Interpolation on a mesh."""

    cdef PetscDMInterpolation dminterp

    def __cinit__(self):
        self.dminterp = NULL

    def __dealloc__(self):
        self.destroy()

    def create(self, comm: Comm | None = None) -> Self:
        """Create a `DMInterpolation` context.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `COMM_SELF`.

        See Also
        --------
        destroy, petsc.DMInterpolationCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_SELF)
        cdef PetscDMInterpolation newdminterp = NULL
        CHKERR(DMInterpolationCreate(ccomm, &newdminterp))
        CHKERR(DMInterpolationDestroy(&self.dminterp))
        self.dminterp = newdminterp
        return self

    def destroy(self) -> Self:
        """Destroy the `DMInterpolation` context.

        Collective.

        See Also
        --------
        create, petsc.DMInterpolationDestroy

        """
        CHKERR(DMInterpolationDestroy(&self.dminterp))
        return self

    def evaluate(self, DM dm, Vec x, Vec v=None) -> Vec:
        """Calculate interpolated field values at the interpolation points.

        Collective.

        Parameters
        ----------
        dm
            The `DM`.
        x
            The local vector containing the field to be interpolated.
        v
            A vector capable of holding the interpolated field values.

        See Also
        --------
        petsc.DMInterpolationEvaluate

        """
        if v is None:
            v = Vec()
        if v.vec == NULL:
            CHKERR(DMInterpolationGetVector(self.dminterp, &v.vec))
        CHKERR(DMInterpolationEvaluate(self.dminterp, dm.dm, x.vec, v.vec))
        return v

    def getCoordinates(self) -> Vec:
        """Return the coordinates of each interpolation point.

        Collective.

        The local vector entries correspond to interpolation points lying on
        this process, according to the associated DM.

        See Also
        --------
        petsc.DMInterpolationGetCoordinates

        """
        cdef Vec coords = Vec()
        CHKERR(DMInterpolationGetCoordinates(self.dminterp, &coords.vec))
        CHKERR(PetscINCREF(coords.obj))
        return coords

    def getDim(self) -> int:
        """Return the spatial dimension of the interpolation context.

        Not collective.

        See Also
        --------
        setDim, petsc.DMInterpolationGetDim

        """
        cdef PetscInt cdim = 0
        CHKERR(DMInterpolationGetDim(self.dminterp, &cdim))
        return toInt(cdim)

    def getDof(self) -> int:
        """Return the number of fields interpolated at a point.

        Not collective.

        See Also
        --------
        setDof, petsc.DMInterpolationGetDof

        """
        cdef PetscInt cdof = 0
        CHKERR(DMInterpolationGetDof(self.dminterp, &cdof))
        return toInt(cdof)

    def setDim(self, dim: int) -> None:
        """Set the spatial dimension for the interpolation context.

        Not collective.

        Parameters
        ----------
        dim
            The spatial dimension.

        See Also
        --------
        getDim, petsc.DMInterpolationSetDim

        """
        cdef PetscInt cdim = asInt(dim)
        CHKERR(DMInterpolationSetDim(self.dminterp, cdim))

    def setDof(self, dof: int) -> None:
        """Set the number of fields interpolated at a point.

        Not collective.

        Parameters
        ----------
        dof
            The number of fields.

        See Also
        --------
        getDof, petsc.DMInterpolationSetDof

        """
        cdef PetscInt cdof = asInt(dof)
        CHKERR(DMInterpolationSetDof(self.dminterp, cdof))

    def setUp(
        self,
        DM dm,
        redundantPoints: bool = False,
        ignoreOutsideDomain: bool = False) -> None:
        """Compute spatial indices for point location during interpolation.

        Collective.

        Parameters
        ----------
        dm
            The DM for the function space used for interpolation.
        redundantPoints
            If `True`, all processes are passing in the same array of points.
            Otherwise, points need to be communicated among processes.
        ignoreOutsideDomain
            Ignore points outside of the domain if `True`; otherwise, return an
            error.

        See Also
        --------
        petsc.DMInterpolationSetUp

        """
        cdef PetscBool credundantPoints = asBool(redundantPoints)
        cdef PetscBool cignoreOutsideDomain = asBool(ignoreOutsideDomain)
        CHKERR(DMInterpolationSetUp(self.dminterp, dm.dm, credundantPoints, cignoreOutsideDomain))

    def getVector(self) -> Vec:
        """Return a `Vec` which can hold all the interpolated field values.

        Collective.

        This vector should be returned using `restoreVector`.

        See Also
        --------
        restoreVector, petsc.DMInterpolationGetVector

        """
        cdef Vec vec = Vec()
        CHKERR(DMInterpolationGetVector(self.dminterp, &vec.vec))
        return vec

    def restoreVector(self, Vec vec) -> None:
        """Restore a Vec which can hold all the interpolated field values.

        Collective.

        Parameters
        ----------
        vec
            A vector capable of holding the interpolated field values.

        See Also
        --------
        getVector, petsc.DMInterpolationRestoreVector

        """
        CHKERR(DMInterpolationRestoreVector(self.dminterp, &vec.vec))
