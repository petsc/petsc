# --------------------------------------------------------------------

cdef class Section(Object):
    """Mapping from integers in a range to unstructured set of integers."""

    def __cinit__(self):
        self.obj = <PetscObject*> &self.sec
        self.sec  = NULL

    def __dealloc__(self):
        CHKERR( PetscSectionDestroy(&self.sec) )
        self.sec = NULL

    def view(self, Viewer viewer=None) -> None:
        """View the section.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` to display the section.

        See Also
        --------
        petsc.PetscSectionView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscSectionView(self.sec, vwr) )

    def destroy(self) -> Self:
        """Destroy a section.

        Not collective.

        See Also
        --------
        petsc.PetscSectionDestroy

        """
        CHKERR( PetscSectionDestroy(&self.sec) )
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Allocate a section and set the map contents to the default.

        Collective.

        Typical calling sequence:
        - `create`
        - `setNumFields`
        - `setChart`
        - `setDof`
        - `setUp`
        - `getOffset`
        - `destroy`

        The `Section` object and methods are intended to be used in the PETSc
        Vec and Mat implementations. The indices returned by the `Section` are
        appropriate for the kind of `Vec` it is associated with. For example,
        if the vector being indexed is a local vector, we call the section a
        local section. If the section indexes a global vector, we call it a
        global section. For parallel vectors, like global vectors, we use
        negative indices to indicate DOFs owned by other processes.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.PetscSectionCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSection newsec = NULL
        CHKERR( PetscSectionCreate(ccomm, &newsec) )
        PetscCLEAR(self.obj); self.sec = newsec
        return self

    def clone(self) -> Section:
        """Return a copy of the section.

        Collective.

        The copy is shallow, if possible.

        See Also
        --------
        petsc.PetscSectionClone

        """
        cdef Section sec = <Section>type(self)()
        CHKERR( PetscSectionClone(self.sec, &sec.sec) )
        return sec

    def setUp(self) -> None:
        """Calculate offsets.

        Not collective.

        Offsets are based on the number of degrees of freedom for each point.

        See Also
        --------
        petsc.PetscSectionSetUp

        """
        CHKERR( PetscSectionSetUp(self.sec) )

    def reset(self) -> None:
        """Free all section data.

        Not collective.

        See Also
        --------
        petsc.PetscSectionReset

        """
        CHKERR( PetscSectionReset(self.sec) )

    def getNumFields(self) -> int:
        """Return the number of fields in a section.

        Not collective.

        Returns ``0`` if no fields were defined.

        See Also
        --------
        setNumFields, petsc.PetscSectionGetNumFields

        """
        cdef PetscInt numFields = 0
        CHKERR( PetscSectionGetNumFields(self.sec, &numFields) )
        return toInt(numFields)

    def setNumFields(self, numFields: int) -> None:
        """Set the number of fields in a section.

        Not collective.

        Parameters
        ----------
        numFields
            The number of fields.

        See Also
        --------
        getNumFields, petsc.PetscSectionSetNumFields

        """
        cdef PetscInt cnumFields = asInt(numFields)
        CHKERR( PetscSectionSetNumFields(self.sec, cnumFields) )

    def getFieldName(self, field: int) -> str:
        """Return the name of a field in the section.

        Not collective.

        Parameters
        ----------
        field
            The field number.

        See Also
        --------
        setFieldName, petsc.PetscSectionGetFieldName

        """
        cdef PetscInt cfield = asInt(field)
        cdef const char *fieldName = NULL
        CHKERR( PetscSectionGetFieldName(self.sec,cfield,&fieldName) )
        return bytes2str(fieldName)

    def setFieldName(self, field: int, fieldName: str) -> None:
        """Set the name of a field in the section.

        Not collective.

        Parameters
        ----------
        field
            The field number.
        fieldName
            The field name.

        See Also
        --------
        getFieldName, petsc.PetscSectionSetFieldName

        """
        cdef PetscInt cfield = asInt(field)
        cdef const char *cname = NULL
        fieldName = str2bytes(fieldName, &cname)
        CHKERR( PetscSectionSetFieldName(self.sec,cfield,cname) )

    def getFieldComponents(self, field: int) -> int:
        """Return the number of field components for the given field.

        Not collective.

        Parameters
        ----------
        field
            The field number.

        See Also
        --------
        setFieldComponents, petsc.PetscSectionGetFieldComponents

        """
        cdef PetscInt cfield = asInt(field), cnumComp = 0
        CHKERR( PetscSectionGetFieldComponents(self.sec,cfield,&cnumComp) )
        return toInt(cnumComp)

    def setFieldComponents(self, field: int, numComp: int) -> None:
        """Set the number of field components for the given field.

        Not collective.

        Parameters
        ----------
        field
            The field number.
        numComp
            The number of field components.

        See Also
        --------
        getFieldComponents, petsc.PetscSectionSetFieldComponents

        """
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumComp = asInt(numComp)
        CHKERR( PetscSectionSetFieldComponents(self.sec,cfield,cnumComp) )

    def getChart(self) -> tuple[int, int]:
        """Return the range in which points (indices) lie for this section.

        Not collective.

        The range is [pStart, pEnd), i.e., from the first point to one past the
        last point.

        See Also
        --------
        petsc.PetscSectionGetChart

        """
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( PetscSectionGetChart(self.sec, &pStart, &pEnd) )
        return toInt(pStart), toInt(pEnd)

    def setChart(self, pStart: int, pEnd: int) -> None:
        """Set the range in which points (indices) lie for this section.

        Not collective.

        The range is [pStart, pEnd), i.e., from the first point to one past the
        last point.

        Parameters
        ----------
        pStart
            The first point.
        pEnd
            One past the last point.

        See Also
        --------
        petsc.PetscSectionSetChart

        """
        cdef PetscInt cStart = asInt(pStart)
        cdef PetscInt cEnd   = asInt(pEnd)
        CHKERR( PetscSectionSetChart(self.sec, cStart, cEnd) )

    def getPermutation(self) -> IS:
        """Return the permutation that was set with `setPermutation`.

        Not collective.

        See Also
        --------
        setPermutation, petsc.PetscSectionGetPermutation

        """
        cdef IS perm = IS()
        CHKERR( PetscSectionGetPermutation(self.sec, &perm.iset))
        PetscINCREF(perm.obj)
        return perm

    def setPermutation(self, IS perm) -> None:
        """Set the permutation for [0, pEnd - pStart).

        Not collective.

        Parameters
        ----------
        perm
            The permutation of points.

        See Also
        --------
        getPermutation, petsc.PetscSectionSetPermutation

        """
        CHKERR( PetscSectionSetPermutation(self.sec, perm.iset))

    def getDof(self, point: int) -> int:
        """Return the number of degrees of freedom for a given point.

        Not collective.

        In a global section, this value will be negative for points not owned
        by this process.

        Parameters
        ----------
        point
            The point.

        See Also
        --------
        setDof, addDof, petsc.PetscSectionGetDof

        """
        cdef PetscInt cpoint = asInt(point), cnumDof = 0
        CHKERR( PetscSectionGetDof(self.sec,cpoint,&cnumDof) )
        return toInt(cnumDof)

    def setDof(self, point: int, numDof: int) -> None:
        """Set the number of degrees of freedom associated with a given point.

        Not collective.

        Parameters
        ----------
        point
            The point.
        numDof
            The number of DOFs.

        See Also
        --------
        getDof, addDof, petsc.PetscSectionSetDof

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionSetDof(self.sec,cpoint,cnumDof) )

    def addDof(self, point: int, numDof: int) -> None:
        """Add ``numDof`` degrees of freedom associated with a given point.

        Not collective.

        Parameters
        ----------
        point
            The point.
        numDof
            The number of additional DOFs.

        See Also
        --------
        setDof, getDof, petsc.PetscSectionAddDof

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionAddDof(self.sec,cpoint,cnumDof) )

    def getFieldDof(self, point: int, field: int) -> int:
        """Return the number of DOFs associated with a field on a given point.

        Not collective.

        Parameters
        ----------
        point
            The point.
        field
            The field.

        See Also
        --------
        setFieldDof, petsc.PetscSectionGetFieldDof

        """
        cdef PetscInt cpoint = asInt(point), cnumDof = 0
        cdef PetscInt cfield = asInt(field)
        CHKERR( PetscSectionGetFieldDof(self.sec,cpoint,cfield,&cnumDof) )
        return toInt(cnumDof)

    def setFieldDof(self, point: int, field: int, numDof: int) -> None:
        """Set the number of DOFs associated with a field on a given point.

        Not collective.

        Parameters
        ----------
        point
            The point.
        field
            The field.
        numDof
            The number of DOFs.

        See Also
        --------
        getFieldDof, addFieldDof, petsc.PetscSectionSetFieldDof

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionSetFieldDof(self.sec,cpoint,cfield,cnumDof) )

    def addFieldDof(self, point: int, field: int, numDof: int) -> None:
        """Add ``numDof`` DOFs associated with a field on a given point.

        Not collective.

        Parameters
        ----------
        point
            The point.
        field
            The field.
        numDof
            The number of additional DOFs.

        See Also
        --------
        setFieldDof, getFieldDof, petsc.PetscSectionAddFieldDof

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionAddFieldDof(self.sec,cpoint,cfield,cnumDof) )

    def getConstraintDof(self, point: int) -> int:
        """Return the number of constrained DOFs associated with a given point.

        Not collective.

        Parameters
        ----------
        point
            The point.

        See Also
        --------
        setConstraintDof, petsc.PetscSectionGetConstraintDof

        """
        cdef PetscInt cpoint = asInt(point), cnumDof = 0
        CHKERR( PetscSectionGetConstraintDof(self.sec,cpoint,&cnumDof) )
        return toInt(cnumDof)

    def setConstraintDof(self, point: int, numDof: int) -> None:
        """Set the number of constrained DOFs associated with a given point.

        Not collective.

        Parameters
        ----------
        point
            The point.
        numDof
            The number of DOFs which are fixed by constraints.

        See Also
        --------
        getConstraintDof, addConstraintDof, petsc.PetscSectionSetConstraintDof

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionSetConstraintDof(self.sec,cpoint,cnumDof) )

    def addConstraintDof(self, point: int, numDof: int) -> None:
        """Increment the number of constrained DOFs for a given point.

        Not collective.

        Parameters
        ----------
        point
            The point.
        numDof
            The number of additional DOFs which are fixed by constraints.

        See Also
        --------
        setConstraintDof, getConstraintDof, petsc.PetscSectionAddConstraintDof

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionAddConstraintDof(self.sec,cpoint,cnumDof) )

    def getFieldConstraintDof(self, point: int, field: int) -> int:
        """Return the number of constrained DOFs for a given field on a point.

        Not collective.

        Parameters
        ----------
        point
            The point.
        field
            The field.

        See Also
        --------
        setFieldConstraintDof, petsc.PetscSectionGetFieldConstraintDof

        """
        cdef PetscInt cpoint = asInt(point), cnumDof = 0
        cdef PetscInt cfield = asInt(field)
        CHKERR( PetscSectionGetFieldConstraintDof(self.sec,cpoint,cfield,&cnumDof) )
        return toInt(cnumDof)

    def setFieldConstraintDof(
        self,
        point: int,
        field: int,
        numDof: int,
    ) -> None:
        """Set the number of constrained DOFs for a given field on a point.

        Not collective.

        Parameters
        ----------
        point
            The point.
        field
            The field.
        numDof
            The number of DOFs which are fixed by constraints.

        See Also
        --------
        getFieldConstraintDof, addFieldConstraintDof
        petsc.PetscSectionSetFieldConstraintDof

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionSetFieldConstraintDof(self.sec,cpoint,cfield,cnumDof) )

    def addFieldConstraintDof(
        self,
        point: int,
        field: int,
        numDof: int,
    ) -> None:
        """Add ``numDof`` constrained DOFs for a given field on a point.

        Not collective.

        Parameters
        ----------
        point
            The point.
        field
            The field.
        numDof
            The number of additional DOFs which are fixed by constraints.

        See Also
        --------
        setFieldConstraintDof, getFieldConstraintDof
        petsc.PetscSectionAddFieldConstraintDof

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt cnumDof = asInt(numDof)
        CHKERR( PetscSectionAddFieldConstraintDof(self.sec,cpoint,cfield,cnumDof) )

    def getConstraintIndices(self, point: int) -> ArrayInt:
        """Return the point DOFs numbers which are constrained for a given point.

        Not collective.

        The range is in [0, DOFs).

        Parameters
        ----------
        point
            The point.

        See Also
        --------
        setConstraintIndices, petsc.PetscSectionGetConstraintIndices

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt nindex = 0
        cdef const PetscInt *indices = NULL
        CHKERR( PetscSectionGetConstraintDof(self.sec, cpoint, &nindex) )
        CHKERR( PetscSectionGetConstraintIndices(self.sec, cpoint, &indices) )
        return array_i(nindex, indices)

    def setConstraintIndices(self, point: int, indices: Sequence[int]) -> None:
        """Set the point DOFs numbers, in [0, DOFs), which are constrained.

        Not collective.

        Parameters
        ----------
        point
            The point.
        indices
            The constrained DOFs.

        See Also
        --------
        getConstraintIndices, petsc.PetscSectionSetConstraintIndices

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt nindex = 0
        cdef PetscInt *cindices = NULL
        indices = iarray_i(indices, &nindex, &cindices)
        CHKERR( PetscSectionSetConstraintDof(self.sec,cpoint,nindex) )
        CHKERR( PetscSectionSetConstraintIndices(self.sec,cpoint,cindices) )

    def getFieldConstraintIndices(self, point: int, field: int) -> ArrayInt:
        """Return the field DOFs numbers, in [0, DOFs), which are constrained.

        Not collective.

        The constrained DOFs are sorted in ascending order.

        Parameters
        ----------
        field
            The field number.
        point
            The point.

        See Also
        --------
        setFieldConstraintIndices, petsc.PetscSectionGetFieldConstraintIndices

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt nindex = 0
        cdef const PetscInt *indices = NULL
        CHKERR( PetscSectionGetFieldConstraintDof(self.sec,cpoint,cfield,&nindex) )
        CHKERR( PetscSectionGetFieldConstraintIndices(self.sec,cpoint,cfield,&indices) )
        return array_i(nindex, indices)

    def setFieldConstraintIndices(
        self,
        point: int,
        field: int,
        indices: Sequence[int],
    ) -> None:
        """Set the field DOFs numbers, in [0, DOFs), which are constrained.

        Not collective.

        Parameters
        ----------
        point
            The point.
        field
            The field number.
        indices
            The constrained DOFs.

        See Also
        --------
        getFieldConstraintIndices, petsc.PetscSectionSetFieldConstraintIndices

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt nindex = 0
        cdef PetscInt *cindices = NULL
        indices = iarray_i(indices, &nindex, &cindices)
        CHKERR( PetscSectionSetFieldConstraintDof(self.sec,cpoint,cfield,nindex) )
        CHKERR( PetscSectionSetFieldConstraintIndices(self.sec,cpoint,cfield,cindices) )

    def getMaxDof(self) -> int:
        """Return the maximum number of DOFs for any point in the section.

        Not collective.

        See Also
        --------
        petsc.PetscSectionGetMaxDof

        """
        cdef PetscInt maxDof = 0
        CHKERR( PetscSectionGetMaxDof(self.sec,&maxDof) )
        return toInt(maxDof)

    def getStorageSize(self) -> int:
        """Return the size capable of holding all the DOFs defined in a section.

        Not collective.

        See Also
        --------
        getConstrainedStorageSize, petsc.PetscSectionGetStorageSize

        """
        cdef PetscInt size = 0
        CHKERR( PetscSectionGetStorageSize(self.sec,&size) )
        return toInt(size)

    def getConstrainedStorageSize(self) -> int:
        """Return the size capable of holding all unconstrained DOFs in a section.

        Not collective.

        See Also
        --------
        getStorageSize, petsc.PetscSectionGetConstrainedStorageSize

        """
        cdef PetscInt size = 0
        CHKERR( PetscSectionGetConstrainedStorageSize(self.sec,&size) )
        return toInt(size)

    def getOffset(self, point: int) -> int:
        """Return the offset for the DOFs associated with the given point.

        Not collective.

        In a global section, this offset will be negative for points not owned
        by this process.

        Parameters
        ----------
        point
            The point.

        See Also
        --------
        setOffset, petsc.PetscSectionGetOffset

        """
        cdef PetscInt cpoint = asInt(point), offset = 0
        CHKERR( PetscSectionGetOffset(self.sec,cpoint,&offset) )
        return toInt(offset)

    def setOffset(self, point: int, offset: int) -> None:
        """Set the offset for the DOFs associated with the given point.

        Not collective.

        The user usually does not call this function, but uses `setUp`.

        Parameters
        ----------
        point
            The point.
        offset
            The offset.

        See Also
        --------
        getOffset, petsc.PetscSectionSetOffset

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt coffset = asInt(offset)
        CHKERR( PetscSectionSetOffset(self.sec,cpoint,coffset) )

    def getFieldOffset(self, point: int, field: int) -> int:
        """Return the offset for the field DOFs on the given point.

        Not collective.

        In a global section, this offset will be negative for points not owned
        by this process.

        Parameters
        ----------
        point
            The point.
        field
            The field.

        See Also
        --------
        setFieldOffset, petsc.PetscSectionGetFieldOffset

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt offset = 0
        CHKERR( PetscSectionGetFieldOffset(self.sec,cpoint,cfield,&offset) )
        return toInt(offset)

    def setFieldOffset(self, point: int, field: int, offset: int) -> None:
        """Set the offset for the DOFs on the given field at a point.

        Not collective.

        The user usually does not call this function, but uses `setUp`.

        Parameters
        ----------
        point
            The point.
        field
            The field.
        offset
            The offset.

        See Also
        --------
        getFieldOffset, petsc.PetscSectionSetFieldOffset

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        cdef PetscInt coffset = asInt(offset)
        CHKERR( PetscSectionSetFieldOffset(self.sec,cpoint,cfield,coffset) )

    def getOffsetRange(self) -> tuple[int,int]:
        """Return the full range of offsets, [start, end), for a section.

        Not collective.

        See Also
        --------
        petsc.PetscSectionGetOffsetRange

        """
        cdef PetscInt oStart = 0, oEnd = 0
        CHKERR( PetscSectionGetOffsetRange(self.sec,&oStart,&oEnd) )
        return toInt(oStart),toInt(oEnd)

    # FIXME: Hardcoded PETSC_FALSE parameters
    def createGlobalSection(self, SF sf) -> Section:
        """Create a section describing the global field layout.

        The section describes the global field layout using the local section
        and an `SF` describing the section point overlap.

        If we have a set of local sections defining the layout of a set of
        local vectors, and also an `SF` to determine which section points are
        shared and the ownership, we can calculate a global section defining
        the parallel data layout, and the associated global vector.

        This gives negative sizes and offsets to points not owned by this
        process.

        ``includeConstraints`` and ``localOffsets`` parameters of the C API
        are always set to `False`.

        Parameters
        ----------
        sf
            The `SF` describing the parallel layout of the section points
            (leaves are unowned local points).

        See Also
        --------
        petsc.PetscSectionCreateGlobalSection

        """
        cdef Section gsec = Section()
        CHKERR( PetscSectionCreateGlobalSection(self.sec,sf.sf,PETSC_FALSE,PETSC_FALSE,&gsec.sec) )
        return gsec
