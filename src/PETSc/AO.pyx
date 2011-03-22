# --------------------------------------------------------------------

class AOType(object):
    BASIC   = S_(AOBASIC)
    MAPPING = S_(AOMAPPING)

# --------------------------------------------------------------------

cdef class AO(Object):

    Type = AOType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.ao
        self.ao = NULL

    def view(self, Viewer viewer=None):
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( AOView(self.ao, cviewer) )

    def destroy(self):
        CHKERR( AODestroy(self.ao) )
        self.ao = NULL
        return self

    def createBasic(self, app, petsc=None, comm=None):
        cdef PetscIS isapp = NULL, ispetsc = NULL
        cdef PetscInt napp = 0, *idxapp = NULL,
        cdef PetscInt npetsc = 0, *idxpetsc = NULL
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscAO newao = NULL
        if isinstance(app, IS):
            isapp = (<IS>app).iset
            if petsc is not None:
                ispetsc = (<IS?>petsc).iset
            CHKERR( AOCreateBasicIS(isapp, ispetsc, &newao) )
        else:
            app = iarray_i(app, &napp, &idxapp)
            if petsc is not None:
                petsc = iarray_i(app, &npetsc, &idxpetsc)
                assert napp == npetsc, "incompatible array sizes"
            CHKERR( AOCreateBasic(ccomm, napp, idxapp, idxpetsc, &newao) )
        PetscCLEAR(self.obj); self.ao = newao
        return self

    def createMapping(self, app, petsc=None, comm=None):
        cdef PetscIS isapp = NULL, ispetsc = NULL
        cdef PetscInt napp = 0, *idxapp = NULL,
        cdef PetscInt npetsc = 0, *idxpetsc = NULL
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscAO newao = NULL
        if isinstance(app, IS):
            isapp = (<IS>app).iset
            if petsc is not None:
                ispetsc = (<IS?>petsc).iset
            CHKERR( AOCreateMappingIS(isapp, ispetsc, &newao) )
        else:
            app = iarray_i(app, &napp, &idxapp)
            if petsc is not None:
                petsc = iarray_i(app, &npetsc, &idxpetsc)
                assert napp == npetsc, "incompatible array sizes"
            CHKERR( AOCreateMapping(ccomm, napp, idxapp, idxpetsc, &newao) )
        PetscCLEAR(self.obj); self.ao = newao
        return self

    def getType(self):
        cdef PetscAOType aotype = AOBASIC
        CHKERR( AOGetType(self.ao, &aotype) )
        return aotype

    def app2petsc(self, indices):
        cdef PetscIS iset = NULL
        cdef PetscInt nidx = 0, *idx = NULL
        if isinstance(indices, IS):
            iset = (<IS>indices).iset
            CHKERR( AOApplicationToPetscIS(self.ao, iset) )
        else:
            indices = oarray_i(indices, &nidx, &idx)
            CHKERR( AOApplicationToPetsc(self.ao, nidx, idx) )
        return indices

    def petsc2app(self, indices):
        cdef PetscIS iset = NULL
        cdef PetscInt nidx = 0, *idx = NULL
        if isinstance(indices, IS):
            iset = (<IS>indices).iset
            CHKERR( AOPetscToApplicationIS(self.ao, iset) )
        else:
            indices = oarray_i(indices, &nidx, &idx)
            CHKERR( AOPetscToApplication(self.ao, nidx, idx) )
        return indices

# --------------------------------------------------------------------

del AOType

# --------------------------------------------------------------------
