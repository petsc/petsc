# --------------------------------------------------------------------

cdef class DMComposite(DM):
    def create(self, comm=None, subdms=[]):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm
        CHKERR( DMCompositeCreate(ccomm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def addDM(self, DM dm not None):
        """Add DM to composite"""
        CHKERR( DMCompositeAddDM(self.dm, dm.dm) )

    def getNumberDM(self):
        cdef PetscInt n
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        return toInt(n)

    def scatter(self, Vec gvec not None, lvecs):
        """Scatter coupled global vector into split local vectors"""
        cdef PetscInt n
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        cdef PetscVec *clvecs = NULL
        cdef object clvecs_mem = oarray_p(empty_p(n), NULL, <void**>&clvecs)
        for i from 0 <= i < n:
            clvecs[i] = (<Vec?>lvecs[i]).vec
        CHKERR( DMCompositeScatterArray(self.dm, gvec.vec, clvecs) )

    def gather(self, Vec gvec not None, imode, lvecs):
        """Gather split local vectors into coupled global vector"""
        cdef PetscInsertMode cimode = insertmode(imode)
        cdef PetscInt n
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        cdef PetscVec *clvecs = NULL
        cdef object clvecs_mem = oarray_p(empty_p(n), NULL, <void**>&clvecs)
        for i from 0 <= i < n:
            clvecs[i] = (<Vec?>lvecs[i]).vec
        CHKERR( DMCompositeGatherArray(self.dm, gvec.vec, cimode, clvecs) )

    def getEntries(self):
        """Get tuple of sub-DMs contained in the DMComposite"""
        cdef PetscInt n
        CHKERR( DMCompositeGetNumberDM(self.dm, &n) )
        cdef PetscDM *cdms
        cdef object cdms_mem = oarray_p(empty_p(n), NULL, <void**>&cdms)
        CHKERR( DMCompositeGetEntriesArray(self.dm, cdms) )
        cdef DM ent = None
        cdef list entries = []
        for i from 0 <= i < n:
            ent = subtype_DM(cdms[i])()
            ent.dm = cdms[i]
            PetscINCREF(ent.obj)
            entries.append(ent)
        return tuple(entries)

    def getAccess(self, Vec gvec not None, locs):
        """Get access to specified parts of global vector.

        Use via 'with' context manager (PEP 343).
        """
        return _DMComposite_access(self, gvec, locs)
