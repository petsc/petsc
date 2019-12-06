
class QuadType(object):
    pass

class QuadOption(object):
    pass


cdef class Quad(Object):

    Type = QuadType
    Option = QuadOption

    def __cinit__(self):
        self.obj = <PetscObject*> &self.quad
        self.quad = NULL

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscQuadrature newquad = NULL
        CHKERR( PetscQuadratureCreate(ccomm, &newquad) )
        PetscCLEAR(self.obj); self.quad = newquad
        return self

    def destroy(self):
        CHKERR( PetscQuadratureDestroy(&self.quad) )
        return self

del QuadType
del QuadOption
