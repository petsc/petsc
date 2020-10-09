# --------------------------------------------------------------------

cdef class DMComposite(DM):

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR( DMCompositeCreate(ccomm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def addDM(self, DM dm, *args):
        """Add DM to composite"""
        CHKERR( DMCompositeAddDM(self.dm, dm.dm) )
        cdef object item
        for item in args:
            dm = <DM?> item
            CHKERR( DMCompositeAddDM(self.dm, dm.dm) )

    def getNumber(self):
        """Get number of sub-DMs contained in the DMComposite"""
        cdef PetscInt n = 0
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        return toInt(n)
    getNumberDM = getNumber

    def getEntries(self):
        """Get tuple of sub-DMs contained in the DMComposite"""
        cdef PetscInt i, n = 0
        cdef PetscDM *cdms = NULL
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        cdef object tmp = oarray_p(empty_p(n), NULL, <void**>&cdms)
        CHKERR( DMCompositeGetEntriesArray(self.dm, cdms) )
        cdef DM entry = None
        cdef list entries = []
        for i from 0 <= i < n:
            entry = subtype_DM(cdms[i])()
            entry.dm = cdms[i]
            PetscINCREF(entry.obj)
            entries.append(entry)
        return tuple(entries)

    def scatter(self, Vec gvec, lvecs):
        """Scatter coupled global vector into split local vectors"""
        cdef PetscInt i, n = 0
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        cdef PetscVec *clvecs = NULL
        cdef object tmp = oarray_p(empty_p(n), NULL, <void**>&clvecs)
        for i from 0 <= i < n:
            clvecs[i] = (<Vec?>lvecs[<Py_ssize_t>i]).vec
        CHKERR( DMCompositeScatterArray(self.dm, gvec.vec, clvecs) )

    def gather(self, Vec gvec, imode, lvecs):
        """Gather split local vectors into coupled global vector"""
        cdef PetscInsertMode cimode = insertmode(imode)
        cdef PetscInt i, n = 0
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        cdef PetscVec *clvecs = NULL
        cdef object tmp = oarray_p(empty_p(n), NULL, <void**>&clvecs)
        for i from 0 <= i < n:
            clvecs[i] = (<Vec?>lvecs[<Py_ssize_t>i]).vec
        CHKERR( DMCompositeGatherArray(self.dm, cimode, gvec.vec, clvecs) )

    def getGlobalISs(self):
        cdef PetscInt i, n = 0
        cdef PetscIS *cis = NULL
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        CHKERR( DMCompositeGetGlobalISs(self.dm, &cis) )
        cdef object isets = [ref_IS(cis[i]) for i from 0 <= i < n]
        for i from 0 <= i < n:
            CHKERR( ISDestroy(&cis[i]) )
        CHKERR( PetscFree(cis) )
        return isets

    def getLocalISs(self):
        cdef PetscInt i, n = 0
        cdef PetscIS *cis = NULL
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        CHKERR( DMCompositeGetLocalISs(self.dm, &cis) )
        cdef object isets = [ref_IS(cis[i]) for i from 0 <= i < n]
        for i from 0 <= i < n:
            CHKERR( ISDestroy(&cis[i]) )
        CHKERR( PetscFree(cis) )
        return isets

    def getLGMaps(self):
        cdef PetscInt i, n = 0
        cdef PetscLGMap *clgm = NULL
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        CHKERR( DMCompositeGetISLocalToGlobalMappings(self.dm, &clgm) )
        cdef object lgms = [ref_LGMap(clgm[i]) for i from 0 <= i < n]
        for i from 0 <= i < n:
            CHKERR( ISLocalToGlobalMappingDestroy(&clgm[i]) )
        CHKERR( PetscFree(clgm) )
        return lgms

    def getAccess(self, Vec gvec, locs=None):
        """Get access to specified parts of global vector.

        Use via 'with' context manager (PEP 343).
        """
        return _DMComposite_access(self, gvec, locs)
