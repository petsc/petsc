# --------------------------------------------------------------------

class FEType(object):
    BASIC     = S_(PETSCFEBASIC)
    OPENCL    = S_(PETSCFEOPENCL)
    COMPOSITE = S_(PETSCFECOMPOSITE)

# --------------------------------------------------------------------

cdef class FE(Object):

    Type = FEType

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

    def createDefault(self, dim, nc, isSimplex, qorder, prefix=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscFE newfe = NULL
        cdef PetscInt cdim = asInt(dim)
        cdef PetscInt cnc = asInt(nc)
        cdef PetscInt cqorder = asInt(qorder)
        cdef PetscBool cisSimplex = asBool(isSimplex)
        cdef const_char *cprefix = NULL
        if prefix:
             prefix = str2bytes(prefix, &cprefix)
        CHKERR( PetscFECreateDefault(ccomm, cdim, cnc, cisSimplex, cprefix, cqorder, &newfe))
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

# --------------------------------------------------------------------

del FEType

# --------------------------------------------------------------------
