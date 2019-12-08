# --------------------------------------------------------------------

class DSType(object):
    BASIC = S_(PETSCDSBASIC)

# --------------------------------------------------------------------

cdef class DS(Object):

    Type = DSType

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.ds
        self.ds  = NULL

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscDSView(self.ds, vwr) )

    def destroy(self):
        CHKERR( PetscDSDestroy(&self.ds) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDS newds = NULL
        CHKERR( PetscDSCreate(ccomm, &newds) )
        PetscCLEAR(self.obj); self.ds = newds
        return self

    def setType(self, ds_type):
        cdef const_char *cval = NULL
        ds_type = str2bytes(ds_type, &cval)
        CHKERR( PetscDSSetType(self.ds, cval) )

    def getType(self):
        cdef PetscDSType cval = NULL
        CHKERR( PetscDSGetType(self.ds, &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( PetscDSSetFromOptions(self.ds) )

    def setUp(self):
        CHKERR( PetscDSSetUp(self.ds) )
        return self

    #

    def getSpatialDimension(self):
        cdef PetscInt dim = 0
        CHKERR( PetscDSGetSpatialDimension(self.ds, &dim) )
        return toInt(dim)

    def getCoordinateDimension(self):
        cdef PetscInt dim = 0
        CHKERR( PetscDSGetCoordinateDimension(self.ds, &dim) )
        return toInt(dim)

    def getNumFields(self):
        cdef PetscInt nf = 0
        CHKERR( PetscDSGetNumFields(self.ds, &nf) )
        return toInt(nf)

    def getFieldIndex(self, Object disc):
        cdef PetscInt field = 0
        CHKERR( PetscDSGetFieldIndex(self.ds, disc.obj[0], &field) )
        return toInt(field)

    def getTotalDimensions(self):
        cdef PetscInt tdim = 0
        CHKERR( PetscDSGetTotalDimension(self.ds, &tdim) )
        return toInt(tdim)

    def getTotalComponents(self):
        cdef PetscInt tcmp = 0
        CHKERR( PetscDSGetTotalComponents(self.ds, &tcmp) )
        return toInt(tcmp)

    def getDimensions(self):
        cdef PetscInt nf = 0, *dims = NULL
        CHKERR( PetscDSGetNumFields(self.ds, &nf) )
        CHKERR( PetscDSGetDimensions(self.ds, &dims) )
        return array_i(nf, dims)

    def getComponents(self):
        cdef PetscInt nf = 0, *cmps = NULL
        CHKERR( PetscDSGetNumFields(self.ds, &nf) )
        CHKERR( PetscDSGetComponents(self.ds, &cmps) )
        return array_i(nf, cmps)

    def setDiscretisation(self, f, disc):
        cdef PetscInt cf = asInt(f)
        cdef FE fe = disc
        CHKERR( PetscDSSetDiscretization(self.ds, cf, <PetscObject> fe.fe) )



# --------------------------------------------------------------------

del DSType

# --------------------------------------------------------------------
