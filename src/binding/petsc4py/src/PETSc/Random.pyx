# --------------------------------------------------------------------

class RandomType(object):
    RAND      = S_(PETSCRAND)
    RAND48    = S_(PETSCRAND48)
    SPRNG     = S_(PETSCSPRNG)
    RANDER48  = S_(PETSCRANDER48)
    RANDOM123 = S_(PETSCRANDOM123)

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
        CHKERR( PetscRandomDestroy(&self.rnd) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        CHKERR( PetscRandomCreate(ccomm, &self.rnd) )
        return self

    def setType(self, rnd_type):
        cdef PetscRandomType cval = NULL
        rnd_type = str2bytes(rnd_type, &cval)
        CHKERR( PetscRandomSetType(self.rnd, cval) )

    def getType(self):
        cdef PetscRandomType cval = NULL
        CHKERR( PetscRandomGetType(self.rnd, &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( PetscRandomSetFromOptions(self.rnd) )

    def getValue(self):
        cdef PetscScalar sval = 0
        CHKERR( PetscRandomGetValue(self.rnd, &sval) )
        return toScalar(sval)

    def getValueReal(self):
        cdef PetscReal rval = 0
        CHKERR( PetscRandomGetValueReal(self.rnd, &rval) )
        return toReal(rval)

    def getSeed(self):
        cdef unsigned long seed = 0
        CHKERR( PetscRandomGetSeed(self.rnd, &seed) )
        return seed

    def setSeed(self, seed=None):
        if seed is not None:
            CHKERR( PetscRandomSetSeed(self.rnd, seed) )
        CHKERR( PetscRandomSeed(self.rnd) )

    def getInterval(self):
        cdef PetscScalar sval1 = 0
        cdef PetscScalar sval2 = 1
        CHKERR( PetscRandomGetInterval(self.rnd, &sval1, &sval2) )
        return (toScalar(sval1), toScalar(sval2))

    def setInterval(self, interval):
        cdef PetscScalar sval1 = 0
        cdef PetscScalar sval2 = 1
        low, high = interval
        sval1 = asScalar(low)
        sval2 = asScalar(high)
        CHKERR( PetscRandomSetInterval(self.rnd, sval1, sval2) )

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

del RandomType

# --------------------------------------------------------------------
