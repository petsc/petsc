
cdef class DMLabel(Object):
    """An object representing a subset of mesh entities from a `DM`."""
    def __cinit__(self):
        self.obj = <PetscObject*> &self.dmlabel
        self.dmlabel  = NULL

    def destroy(self) -> Self:
        """Destroy the label.

        Collective.

        See Also
        --------
        petsc.DMLabelDestroy

        """
        CHKERR(DMLabelDestroy(&self.dmlabel))
        return self

    def view(self, Viewer viewer=None) -> None:
        """View the label.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` to display the graph.

        See Also
        --------
        petsc.DMLabelView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR(DMLabelView(self.dmlabel, vwr))

    def create(self, name: str, comm: Comm | None = None) -> Self:
        """Create a `DMLabel` object, which is a multimap.

        Collective.

        Parameters
        ----------
        name
            The label name.
        comm
            MPI communicator, defaults to `COMM_SELF`.

        See Also
        --------
        petsc.DMLabelCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_SELF)
        cdef PetscDMLabel newdmlabel = NULL
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR(DMLabelCreate(ccomm, cname, &newdmlabel))
        CHKERR(PetscCLEAR(self.obj)); self.dmlabel = newdmlabel
        return self

    def duplicate(self) -> DMLabel:
        """Duplicate the `DMLabel`.

        Collective.

        See Also
        --------
        petsc.DMLabelDuplicate

        """
        cdef DMLabel new = DMLabel()
        CHKERR(DMLabelDuplicate(self.dmlabel, &new.dmlabel))
        return new

    def reset(self) -> None:
        """Destroy internal data structures in the `DMLabel`.

        Not collective.

        See Also
        --------
        petsc.DMLabelReset

        """
        CHKERR(DMLabelReset(self.dmlabel))

    def insertIS(self, IS iset, value: int) -> Self:
        """Set all points in the `IS` to a value.

        Not collective.

        Parameters
        ----------
        iset
            The point IS.
        value
            The point value.

        See Also
        --------
        petsc.DMLabelInsertIS

        """
        cdef PetscInt cvalue = asInt(value)
        CHKERR(DMLabelInsertIS(self.dmlabel, iset.iset, cvalue))
        return self

    def setValue(self, point: int, value: int) -> None:
        """Set the value a label assigns to a point.

        Not collective.

        If the value is the same as the label's default value (which is
        initially ``-1``, and can be changed with `setDefaultValue`), this
        function will do nothing.

        Parameters
        ----------
        point
            The point.
        value
            The point value.

        See Also
        --------
        getValue, setDefaultValue, petsc.DMLabelSetValue

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cvalue = asInt(value)
        CHKERR(DMLabelSetValue(self.dmlabel, cpoint, cvalue))

    def getValue(self, point: int) -> int:
        """Return the value a label assigns to a point.

        Not collective.

        If no value was assigned, a default value will be returned
        The default value, initially ``-1``, can be changed with
        `setDefaultValue`.

        Parameters
        ----------
        point
            The point.

        See Also
        --------
        setValue, setDefaultValue, petsc.DMLabelGetValue

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cvalue = 0
        CHKERR(DMLabelGetValue(self.dmlabel, cpoint, &cvalue))
        return toInt(cvalue)

    def getDefaultValue(self) -> int:
        """Return the default value returned by `getValue`.

        Not collective.

        The default value is returned if a point has not been explicitly given
        a value. When a label is created, it is initialized to ``-1``.

        See Also
        --------
        setDefaultValue, petsc.DMLabelGetDefaultValue

        """
        cdef PetscInt cvalue = 0
        CHKERR(DMLabelGetDefaultValue(self.dmlabel, &cvalue))
        return toInt(cvalue)

    def setDefaultValue(self, value: int) -> None:
        """Set the default value returned by `getValue`.

        Not collective.

        The value is used if a point has not been explicitly given a value.
        When a label is created, the default value is initialized to ``-1``.

        Parameters
        ----------
        value
            The default value.

        See Also
        --------
        getDefaultValue, petsc.DMLabelSetDefaultValue

        """
        cdef PetscInt cvalue = asInt(value)
        CHKERR(DMLabelSetDefaultValue(self.dmlabel, cvalue))

    def clearValue(self, point: int, value: int) -> None:
        """Clear the value a label assigns to a point.

        Not collective.

        Parameters
        ----------
        point
            The point.
        value
            The point value.

        See Also
        --------
        petsc.DMLabelClearValue

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cvalue = asInt(value)
        CHKERR(DMLabelClearValue(self.dmlabel, cpoint, cvalue))

    def addStratum(self, value: int) -> None:
        """Add a new stratum value in a `DMLabel`.

        Not collective.

        Parameters
        ----------
        value
            The stratum value.

        See Also
        --------
        addStrata, addStrataIS, petsc.DMLabelAddStratum

        """
        cdef PetscInt cvalue = asInt(value)
        CHKERR(DMLabelAddStratum(self.dmlabel, cvalue))

    def addStrata(self, strata: Sequence[int]) -> None:
        """Add new stratum values in a `DMLabel`.

        Not collective.

        Parameters
        ----------
        strata
            The stratum values.

        See Also
        --------
        addStrataIS, addStratum, petsc.DMLabelAddStrata

        """
        cdef PetscInt *istrata = NULL
        cdef PetscInt numStrata = 0
        strata = iarray_i(strata, &numStrata, &istrata)
        CHKERR(DMLabelAddStrata(self.dmlabel, numStrata, istrata))

    def addStrataIS(self, IS iset) -> None:
        """Add new stratum values in a `DMLabel`.

        Not collective.

        Parameters
        ----------
        iset
            Index set with stratum values.

        See Also
        --------
        addStrata, addStratum, petsc.DMLabelAddStrataIS

        """
        CHKERR(DMLabelAddStrataIS(self.dmlabel, iset.iset))

    def getNumValues(self) -> int:
        """Return the number of values that the `DMLabel` takes.

        Not collective.

        See Also
        --------
        petsc.DMLabelGetNumValues

        """
        cdef PetscInt numValues = 0
        CHKERR(DMLabelGetNumValues(self.dmlabel, &numValues))
        return toInt(numValues)

    def getValueIS(self) -> IS:
        """Return an `IS` of all values that the `DMLabel` takes.

        Not collective.

        See Also
        --------
        petsc.DMLabelGetValueIS

        """
        cdef IS iset = IS()
        CHKERR(DMLabelGetValueIS(self.dmlabel, &iset.iset))
        return iset

    def stratumHasPoint(self, value: int, point: int) -> bool:
        """Return whether the stratum contains a point.

        Not collective.

        Parameters
        ----------
        value
            The stratum value.
        point
            The point.

        See Also
        --------
        petsc.DMLabelStratumHasPoint

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cvalue = asInt(value)
        cdef PetscBool ccontains = PETSC_FALSE
        CHKERR(DMLabelStratumHasPoint(self.dmlabel, cvalue, cpoint, &ccontains))
        return toBool(ccontains)

    def hasStratum(self, value: int) -> bool:
        """Determine whether points exist with the given value.

        Not collective.

        Parameters
        ----------
        value
            The stratum value.

        See Also
        --------
        petsc.DMLabelHasStratum

        """
        cdef PetscInt cvalue = asInt(value)
        cdef PetscBool cexists = PETSC_FALSE
        CHKERR(DMLabelHasStratum(self.dmlabel, cvalue, &cexists))
        return toBool(cexists)

    def getStratumSize(self, stratum: int) -> int:
        """Return the size of a stratum.

        Not collective.

        Parameters
        ----------
        stratum
            The stratum value.

        See Also
        --------
        petsc.DMLabelGetStratumSize

        """
        cdef PetscInt cstratum = asInt(stratum)
        cdef PetscInt csize = 0
        CHKERR(DMLabelGetStratumSize(self.dmlabel, cstratum, &csize))
        return toInt(csize)

    def getStratumIS(self, stratum: int) -> IS:
        """Return an `IS` with the stratum points.

        Not collective.

        Parameters
        ----------
        stratum
            The stratum value.

        See Also
        --------
        setStratumIS, petsc.DMLabelGetStratumIS

        """
        cdef PetscInt cstratum = asInt(stratum)
        cdef IS iset = IS()
        CHKERR(DMLabelGetStratumIS(self.dmlabel, cstratum, &iset.iset))
        return iset

    def setStratumIS(self, stratum: int, IS iset) -> None:
        """Set the stratum points using an `IS`.

        Not collective.

        Parameters
        ----------
        stratum
            The stratum value.
        iset
            The stratum points.

        See Also
        --------
        getStratumIS, petsc.DMLabelSetStratumIS

        """
        cdef PetscInt cstratum = asInt(stratum)
        CHKERR(DMLabelSetStratumIS(self.dmlabel, cstratum, iset.iset))

    def clearStratum(self, stratum: int) -> None:
        """Remove a stratum.

        Not collective.

        Parameters
        ----------
        stratum
            The stratum value.

        See Also
        --------
        petsc.DMLabelClearStratum

        """
        cdef PetscInt cstratum = asInt(stratum)
        CHKERR(DMLabelClearStratum(self.dmlabel, cstratum))

    def computeIndex(self) -> None:
        """Create an index structure for membership determination.

        Not collective.

        Automatically determines the bounds.

        See Also
        --------
        petsc.DMLabelComputeIndex

        """
        CHKERR(DMLabelComputeIndex(self.dmlabel))

    def createIndex(self, pStart: int, pEnd: int) -> None:
        """Create an index structure for membership determination.

        Not collective.

        Parameters
        ----------
        pStart
            The smallest point.
        pEnd
            The largest point + 1.

        See Also
        --------
        destroyIndex, petsc.DMLabelCreateIndex

        """
        cdef PetscInt cpstart = asInt(pStart), cpend = asInt(pEnd)
        CHKERR(DMLabelCreateIndex(self.dmlabel, cpstart, cpend))

    def destroyIndex(self) -> None:
        """Destroy the index structure.

        Not collective.

        See Also
        --------
        createIndex, petsc.DMLabelDestroyIndex

        """
        CHKERR(DMLabelDestroyIndex(self.dmlabel))

    def hasValue(self, value: int) -> bool:
        """Determine whether a label assigns the value to any point.

        Not collective.

        Parameters
        ----------
        value
            The value.

        See Also
        --------
        hasPoint, petsc.DMLabelHasValue

        """
        cdef PetscInt cvalue = asInt(value)
        cdef PetscBool cexists = PETSC_FALSE
        CHKERR(DMLabelHasValue(self.dmlabel, cvalue, &cexists))
        return toBool(cexists)

    def hasPoint(self, point: int) -> bool:
        """Determine whether the label contains a point.

        Not collective.

        The user must call `createIndex` before this function.

        Parameters
        ----------
        point
            The point.

        See Also
        --------
        hasValue, petsc.DMLabelHasPoint

        """
        cdef PetscInt cpoint = asInt(point)
        cdef PetscBool cexists = PETSC_FALSE
        CHKERR(DMLabelHasPoint(self.dmlabel, cpoint, &cexists))
        return toBool(cexists)

    def getBounds(self) -> tuple[int, int]:
        """Return the smallest and largest point in the label.

        Not collective.

        The returned values are the smallest point and the largest point + 1.

        See Also
        --------
        petsc.DMLabelGetBounds

        """
        cdef PetscInt cpstart = 0, cpend = 0
        CHKERR(DMLabelGetBounds(self.dmlabel, &cpstart, &cpend))
        return toInt(cpstart), toInt(cpend)

    def filter(self, start: int, end: int) -> None:
        """Remove all points outside of [start, end).

        Not collective.

        Parameters
        ----------
        start
            The first point kept.
        end
            One more than the last point kept.

        See Also
        --------
        petsc.DMLabelFilter

        """
        cdef PetscInt cstart = asInt(start), cend = asInt(end)
        CHKERR(DMLabelFilter(self.dmlabel, cstart, cend))

    def permute(self, IS permutation) -> DMLabel:
        """Create a new label with permuted points.

        Not collective.

        Parameters
        ----------
        permutation
            The point permutation.

        See Also
        --------
        petsc.DMLabelPermute

        """
        cdef DMLabel new = DMLabel()
        CHKERR(DMLabelPermute(self.dmlabel, permutation.iset, &new.dmlabel))
        return new

    def distribute(self, SF sf) -> DMLabel:
        """Create a new label pushed forward over the `SF`.

        Collective.

        Parameters
        ----------
        sf
            The map from old to new distribution.

        See Also
        --------
        gather, petsc.DMLabelDistribute

        """
        cdef DMLabel new = DMLabel()
        CHKERR(DMLabelDistribute(self.dmlabel, sf.sf, &new.dmlabel))
        return new

    def gather(self, SF sf) -> DMLabel:
        """Gather all label values from leaves into roots.

        Collective.

        This is the inverse operation to `distribute`.

        Parameters
        ----------
        sf
            The `SF` communication map.

        See Also
        --------
        distribute, petsc.DMLabelGather

        """
        cdef DMLabel new = DMLabel()
        CHKERR(DMLabelGather(self.dmlabel, sf.sf, &new.dmlabel))
        return new

    def convertToSection(self) -> tuple[Section, IS]:
        """Return a `Section` and `IS` that encode the label.

        Not collective.

        See Also
        --------
        petsc.DMLabelConvertToSection

        """
        cdef Section section = Section()
        cdef IS iset = IS()
        CHKERR(DMLabelConvertToSection(self.dmlabel, &section.sec, &iset.iset))
        return section, iset

    def getNonEmptyStratumValuesIS(self) -> IS:
        """Return an `IS` of all values that the `DMLabel` takes.

        Not collective.

        See Also
        --------
        petsc.DMLabelGetNonEmptyStratumValuesIS

        """
        cdef IS iset = IS()
        CHKERR(DMLabelGetNonEmptyStratumValuesIS(self.dmlabel, &iset.iset))
        return iset
