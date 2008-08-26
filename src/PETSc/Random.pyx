# --------------------------------------------------------------------

class RandomType(object):
    RAND   = PETSCRAND
    RAND48 = PETSCRAND48
    SPRNG  = PETSCSPRNG

# --------------------------------------------------------------------

cdef class Random(Object):

    Type = RandomType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.rnd
        self.rnd = NULL

    def __call__(self):
        return self.getValue()

    def view(self, Viewer viewer=None):
        assert self.obj != NULL
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscRandomView(self.rnd, vwr) )

    def destroy(self):
        CHKERR( PetscRandomDestroy(self.rnd) )
        self.rnd = NULL
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        CHKERR( PetscRandomCreate(ccomm, &self.rnd) )
        return self

    def setType(self, rnd_type):
        CHKERR( PetscRandomSetType(self.rnd, str2cp(rnd_type)) )

    def getType(self):
        cdef PetscRandomType rnd_type = NULL
        CHKERR( PetscRandomGetType(self.rnd, &rnd_type) )
        return cp2str(rnd_type)

    def setFromOptions(self):
        CHKERR( PetscRandomSetFromOptions(self.rnd) )

    def getValue(self):
        cdef PetscScalar sval
        CHKERR( PetscRandomGetValue(self.rnd, &sval) )
        return sval

    def getValueReal(self):
        cdef PetscReal rval = 0
        CHKERR( PetscRandomGetValueReal(self.rnd, &rval) )
        return rval

    def getValueImaginary(self):
        cdef PetscScalar sval = 0
        CHKERR( PetscRandomGetValueImaginary(self.rnd, &sval) )
        return sval

    def getSeed(self):
        cdef unsigned long seed = 0
        CHKERR( PetscRandomGetSeed(self.rnd, &seed) )
        return seed

    def setSeed(self, seed=None):
        if seed is not None:
            CHKERR( PetscRandomSetSeed(self.rnd, seed) )
        CHKERR( PetscRandomSeed(self.rnd) )

    def getInterval(self):
        cdef PetscScalar low = 0, high = 1
        CHKERR( PetscRandomGetInterval(self.rnd, &low, &high) )
        return (low, high)

    def setInterval(self, interval):
        cdef PetscScalar low = 0, high = 1
        low, high = interval
        CHKERR( PetscRandomSetInterval(self.rnd, low, high) )

    #

    property seed:
        def __get__(self):
            return self.getSeed()
        def __set__(self, value):
            self.setSeed(value)

    property interval:
        def __get__(self):
            return self.getInterval()
        def __set__(self, value):
            self.setInterval(value)

# --------------------------------------------------------------------

