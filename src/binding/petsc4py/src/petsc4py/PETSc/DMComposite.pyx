# --------------------------------------------------------------------

cdef class DMComposite(DM):
    """A DM object that is used to manage data for a collection of DMs."""

    def create(self, comm: Comm | None = None) -> Self:
        """Create a composite object.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.DMCompositeCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR(DMCompositeCreate(ccomm, &newdm))
        CHKERR(PetscCLEAR(self.obj))
        self.dm = newdm
        return self

    def addDM(self, DM dm, *args: DM) -> None:
        """Add a DM vector to the composite.

        Collective.

        Parameters
        ----------
        dm
            The DM object.
        *args
            Additional DM objects.

        See Also
        --------
        petsc.DMCompositeAddDM

        """
        CHKERR(DMCompositeAddDM(self.dm, dm.dm))
        cdef object item
        for item in args:
            dm = <DM?> item
            CHKERR(DMCompositeAddDM(self.dm, dm.dm))

    def getNumber(self) -> int:
        """Get number of sub-DMs contained in the composite.

        Not collective.

        See Also
        --------
        petsc.DMCompositeGetNumberDM

        """
        cdef PetscInt n = 0
        CHKERR(DMCompositeGetNumberDM(self.dm, &n))
        return toInt(n)
    getNumberDM = getNumber

    def getEntries(self) -> list[DM]:
        """Return sub-DMs contained in the composite.

        Not collective.

        See Also
        --------
        petsc.DMCompositeGetEntriesArray

        """
        cdef PetscInt i, n = 0
        cdef PetscDM *cdms = NULL
        CHKERR(DMCompositeGetNumberDM(self.dm, &n))
        cdef object unused = oarray_p(empty_p(<PetscInt>n), NULL, <void**>&cdms)
        CHKERR(DMCompositeGetEntriesArray(self.dm, cdms))
        cdef DM entry = None
        cdef list entries = []
        for i from 0 <= i < n:
            entry = subtype_DM(cdms[i])()
            entry.dm = cdms[i]
            CHKERR(PetscINCREF(entry.obj))
            entries.append(entry)
        return entries

    def scatter(self, Vec gvec, lvecs: Sequence[Vec]) -> None:
        """Scatter coupled global vector into split local vectors.

        Collective.

        Parameters
        ----------
        gvec
            The global vector.
        lvecs
            Array of local vectors.

        See Also
        --------
        gather, petsc.DMCompositeScatterArray

        """
        cdef PetscInt i, n = 0
        CHKERR(DMCompositeGetNumberDM(self.dm, &n))
        cdef PetscVec *clvecs = NULL
        cdef object unused = oarray_p(empty_p(<PetscInt>n), NULL, <void**>&clvecs)
        for i from 0 <= i < n:
            clvecs[i] = (<Vec?>lvecs[<Py_ssize_t>i]).vec
        CHKERR(DMCompositeScatterArray(self.dm, gvec.vec, clvecs))

    def gather(self, Vec gvec, imode: InsertModeSpec, lvecs: Sequence[Vec]) -> None:
        """Gather split local vectors into a coupled global vector.

        Collective.

        Parameters
        ----------
        gvec
            The global vector.
        imode
            The insertion mode.
        lvecs
            The individual sequential vectors.

        See Also
        --------
        scatter, petsc.DMCompositeGatherArray

        """
        cdef PetscInsertMode cimode = insertmode(imode)
        cdef PetscInt i, n = 0
        CHKERR(DMCompositeGetNumberDM(self.dm, &n))
        cdef PetscVec *clvecs = NULL
        cdef object unused = oarray_p(empty_p(<PetscInt>n), NULL, <void**>&clvecs)
        for i from 0 <= i < n:
            clvecs[i] = (<Vec?>lvecs[<Py_ssize_t>i]).vec
        CHKERR(DMCompositeGatherArray(self.dm, cimode, gvec.vec, clvecs))

    def getGlobalISs(self) -> list[IS]:
        """Return the index sets for each composed object in the composite.

        Collective.

        These could be used to extract a subset of vector entries for a
        "multi-physics" preconditioner.

        Use `getLocalISs` for index sets in the packed local numbering, and
        `getLGMaps` for to map local sub-DM (including ghost) indices to packed
        global indices.

        See Also
        --------
        petsc.DMCompositeGetGlobalISs

        """
        cdef PetscInt i, n = 0
        cdef PetscIS *cis = NULL
        CHKERR(DMCompositeGetNumberDM(self.dm, &n))
        CHKERR(DMCompositeGetGlobalISs(self.dm, &cis))
        cdef object isets = [ref_IS(cis[i]) for i from 0 <= i < n]
        for i from 0 <= i < n:
            CHKERR(ISDestroy(&cis[i]))
        CHKERR(PetscFree(cis))
        return isets

    def getLocalISs(self) -> list[IS]:
        """Return index sets for each component of a composite local vector.

        Not collective.

        To get the composite global indices at all local points (including
        ghosts), use `getLGMaps`.

        To get index sets for pieces of the composite global vector, use
        `getGlobalISs`.

        See Also
        --------
        petsc.DMCompositeGetLocalISs

        """
        cdef PetscInt i, n = 0
        cdef PetscIS *cis = NULL
        CHKERR(DMCompositeGetNumberDM(self.dm, &n))
        CHKERR(DMCompositeGetLocalISs(self.dm, &cis))
        cdef object isets = [ref_IS(cis[i]) for i from 0 <= i < n]
        for i from 0 <= i < n:
            CHKERR(ISDestroy(&cis[i]))
        CHKERR(PetscFree(cis))
        return isets

    def getLGMaps(self) -> list[LGMap]:
        """Return a local-to-global mapping for each DM in the composite.

        Collective.

        Note that this includes all the ghost points that individual ghosted
        DMDA may have.

        See Also
        --------
        petsc.DMCompositeGetISLocalToGlobalMappings

        """
        cdef PetscInt i, n = 0
        cdef PetscLGMap *clgm = NULL
        CHKERR(DMCompositeGetNumberDM(self.dm, &n))
        CHKERR(DMCompositeGetISLocalToGlobalMappings(self.dm, &clgm))
        cdef object lgms = [ref_LGMap(clgm[i]) for i from 0 <= i < n]
        for i from 0 <= i < n:
            CHKERR(ISLocalToGlobalMappingDestroy(&clgm[i]))
        CHKERR(PetscFree(clgm))
        return lgms

    def getAccess(self, Vec gvec, locs: Sequence[int] | None = None) -> Any:
        """Get access to the individual vectors from the global vector.

        Not collective.

        Use via `with` context manager (PEP 343).

        Parameters
        ----------
        gvec
            The global vector.
        locs
            Indices of vectors wanted, or `None` to get all vectors.

        """
        return _DMComposite_access(self, gvec, locs)
