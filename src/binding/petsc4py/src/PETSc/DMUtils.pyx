
cdef class DMInterpolation:

    cdef PetscDMInterpolation dminterp

    def __cinit__(self):
        self.dminterp = NULL

    def __dealloc__(self):
        self.destroy()

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_SELF)
        cdef PetscDMInterpolation new = NULL
        CHKERR( DMInterpolationCreate(ccomm, &new) )
        self.dminterp = new

    def destroy(self):
        CHKERR( DMInterpolationDestroy(&self.dminterp))

    def evaluate(self, DM dm, Vec x):
        cdef Vec v = Vec()
        CHKERR( DMInterpolationEvaluate(self.dminterp, dm.dm, x.vec, v.vec ) )
        return v

    def getCoordinates(self):
        cdef Vec coords = Vec()
        CHKERR( DMInterpolationGetCoordinates(self.dminterp, &coords.vec) )
        return coords

    def getDim(self):
        cdef PetscInt cdim = 0
        CHKERR( DMInterpolationGetDim(self.dminterp, &cdim) )
        return toInt(cdim)
    
    def getDof(self):
        cdef PetscInt cdof = 0
        CHKERR( DMInterpolationGetDof(self.dminterp, &cdof) )
        return toInt(cdof)

    def setDim(self, dim):
        cdef PetscInt cdim = asInt(dim)
        CHKERR( DMInterpolationSetDim(self.dminterp, cdim) )
    
    def setDof(self, dof):
        cdef PetscInt cdof = asInt(dof)
        CHKERR( DMInterpolationSetDof(self.dminterp, cdof) )

    def setUp(self, DM dm, redundantPoints=False, ignoreOutsideDomain=False):
        cdef PetscBool credundantPoints = asBool(redundantPoints)
        cdef PetscBool cignoreOutsideDomain = asBool(ignoreOutsideDomain)
        CHKERR( DMInterpolationSetUp(self.dminterp, dm.dm, credundantPoints, cignoreOutsideDomain) )

    def getVector(self):
        cdef Vec vec = Vec()
        CHKERR( DMInterpolationGetVector(self.dminterp, &vec.vec))
        return vec

    def restoreVector(self, Vec vec):
        CHKERR( DMInterpolationRestoreVector(self.dminterp, &vec.vec) )
        return vec