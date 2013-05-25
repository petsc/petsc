# --------------------------------------------------------------------

cdef extern from * nogil:

    int DMCompositeCreate(MPI_Comm,PetscDM*)
    int DMCompositeAddDM(PetscDM,PetscDM)
    int DMCompositeGetNumberDM(PetscDM,PetscInt*)
    int DMCompositeScatterArray(PetscDM,PetscVec,PetscVec*)
    int DMCompositeGatherArray(PetscDM,PetscVec,PetscInsertMode,PetscVec*)
    int DMCompositeGetEntriesArray(PetscDM,PetscDM*)
    int DMCompositeGetAccessArray(PetscDM,PetscVec,PetscInt,const_PetscInt*,PetscVec*)
    int DMCompositeRestoreAccessArray(PetscDM,PetscVec,PetscInt,const_PetscInt*,PetscVec*)


cdef class _DMComposite_access:
    cdef PetscDM  dm
    cdef PetscVec gvec
    cdef PetscInt nlocs
    cdef PetscInt *locs
    cdef PetscVec *vecs
    cdef object locs_mem
    cdef object vecs_mem

    def __cinit__(self, DM dm, Vec gvec not None, locs=None):
        self.dm = dm.dm
        CHKERR( PetscINCREF(<PetscObject*>&self.dm) )
        self.gvec = gvec.vec
        CHKERR( PetscINCREF(<PetscObject*>&self.gvec) )
        if locs is None:
            CHKERR( DMCompositeGetNumberDM(self.dm, &self.nlocs) )
            locs = arange(0, <long>self.nlocs, 1)
        self.locs_mem = iarray_i(locs, &self.nlocs, &self.locs)
        self.vecs_mem = oarray_p(empty_p(self.nlocs), NULL, <void**>&self.vecs)

    def __dealloc__(self):
        CHKERR( DMDestroy(&self.dm) )
        CHKERR( VecDestroy(&self.gvec) )

    def __enter__(self):
        CHKERR( DMCompositeGetAccessArray(self.dm, self.gvec, self.nlocs, self.locs, self.vecs) )
        cdef list access = []
        cdef Vec x
        for i from 0 <= i < self.nlocs:
            x = Vec()
            x.vec = self.vecs[i]
            PetscINCREF(x.obj)
            access.append(x)
        return tuple(access)

    def __exit__(self, *exc):
        CHKERR( DMCompositeRestoreAccessArray(self.dm, self.gvec, self.nlocs, self.locs, self.vecs) )
        self.nlocs = 0
        self.locs = NULL
        self.vecs = NULL
        self.locs_mem = None
        self.vecs_mem = None
