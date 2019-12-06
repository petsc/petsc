
class FEType(object):
    BASIC     = S_(PETSCFEBASIC)
    OPENCL    = S_(PETSCFEOPENCL)
    COMPOSITE = S_(PETSCFECOMPOSITE)

class FEOption(object):
    pass


cdef class FE(Object):

    Type = FEType
    Option = FEOption

    def __cinit__(self):
        self.obj = <PetscObject*> &self.fe
        self.fe = NULL

    def destroy(self):
        CHKERR( PetscFEDestroy(&self.fe) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscFE newfe = NULL
        CHKERR( PetscFECreate(ccomm, &newfe) )
        PetscCLEAR(self.obj); self.fe = newfe
        return self

    def createDefault(self, dim, Nc, isSimplex, qorder, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscFE newfe = NULL
        CHKERR( PetscFECreateDefault(ccomm, dim, Nc, isSimplex, NULL, qorder, &newfe))
        PetscCLEAR(self.obj); self.fe = newfe
        return self

    def getQuadrature(self):
        cdef Quad quad = Quad()
        CHKERR( PetscFEGetQuadrature(self.fe, &quad.quad) )
        return quad

    def getFaceQuadrature(self):
        cdef Quad quad = Quad()
        CHKERR( PetscFEGetFaceQuadrature(self.fe, &quad.quad) )
        return quad

    def setQuadrature(self, Quad quad):
        CHKERR( PetscFESetQuadrature(self.fe, quad.quad) )
        return self

    def setFaceQuadrature(self, Quad quad):
        CHKERR( PetscFESetFaceQuadrature(self.fe, quad.quad) )
        return self

    def setType(self, typeFE):
        CHKERR( PetscFESetType(self.fe, typeFE) )
        return self

del FEType
del FEOption
