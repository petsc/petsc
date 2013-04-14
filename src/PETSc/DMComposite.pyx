# --------------------------------------------------------------------

cdef class DMComposite(DM):

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR( DMCompositeCreate(ccomm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def addDM(self, DM dm not None, *args):
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
        cdef PetscInt n = 0
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        cdef PetscDM *cdms = NULL
        cdef object cdms_mem = oarray_p(empty_p(n), NULL, <void**>&cdms)
        CHKERR( DMCompositeGetEntriesArray(self.dm, cdms) )
        cdef DM entry = None
        cdef list entries = []
        for i from 0 <= i < n:
            entry = subtype_DM(cdms[i])()
            entry.dm = cdms[i]
            PetscINCREF(entry.obj)
            entries.append(entry)
        return tuple(entries)

    def scatter(self, Vec gvec not None, lvecs):
        """Scatter coupled global vector into split local vectors"""
        cdef PetscInt n = 0
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        cdef PetscVec *clvecs = NULL
        cdef object clvecs_mem = oarray_p(empty_p(n), NULL, <void**>&clvecs)
        for i from 0 <= i < n:
            clvecs[i] = (<Vec?>lvecs[i]).vec
        CHKERR( DMCompositeScatterArray(self.dm, gvec.vec, clvecs) )

    def gather(self, Vec gvec not None, imode, lvecs):
        """Gather split local vectors into coupled global vector"""
        cdef PetscInsertMode cimode = insertmode(imode)
        cdef PetscInt n = 0
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        cdef PetscVec *clvecs = NULL
        cdef object clvecs_mem = oarray_p(empty_p(n), NULL, <void**>&clvecs)
        for i from 0 <= i < n:
            clvecs[i] = (<Vec?>lvecs[i]).vec
        CHKERR( DMCompositeGatherArray(self.dm, gvec.vec, cimode, clvecs) )

    def getAccess(self, Vec gvec not None, locs):
        """Get access to specified parts of global vector.

        Use via 'with' context manager (PEP 343).
        """
        return _DMComposite_access(self, gvec, locs)
