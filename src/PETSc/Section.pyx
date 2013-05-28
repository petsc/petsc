# --------------------------------------------------------------------

cdef class Section(Object):

    def __cinit__(self):
        if PETSC_VERSION_GT(3,3,0):
            self.obj = <PetscObject*> &self.sec
        self.sec  = NULL

    def __dealloc__(self):
        CHKERR( PetscSectionDestroy(&self.sec) )
        self.sec = NULL

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscSectionView(self.sec, vwr) )

    def destroy(self):
        CHKERR( PetscSectionDestroy(&self.sec) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSection newsec = NULL
        CHKERR( PetscSectionCreate(ccomm, &newsec) )
        PetscCLEAR(self.obj); self.sec = newsec
        return self

    def clone(self):
        cdef Section sec = <Section>type(self)()
        CHKERR( PetscSectionClone(self.sec, &sec.sec) )
        return sec
    
    def setUp(self):
        CHKERR( PetscSectionSetUp(self.sec) )

    def reset(self):
        CHKERR( PetscSectionReset(self.sec) )

