# --------------------------------------------------------------------

cdef class Quad(Object):

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

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscQuadratureView(self.quad, vwr) )

    def getData(self):
        cdef PetscInt cdim = 0 
        cdef PetscInt cnc = 0
        cdef PetscInt cnpoints = 0
        cdef const PetscReal *cpoints = NULL
        cdef const PetscReal *cweights = NULL
        CHKERR( PetscQuadratureGetData(self.quad, &cdim, &cnc, &cnpoints, &cpoints, &cweights))
        return array_r(cnpoints*cdim, cpoints), array_r(cnpoints*cnc, cweights)
    
    def getNumComponents(self):
        cdef PetscInt cnc = 0
        CHKERR( PetscQuadratureGetNumComponents(self.quad, &cnc) )
        return toInt(cnc)

    def setNumComponents(self, nc):
        cdef PetscInt cnc = asInt(nc)
        CHKERR( PetscQuadratureSetNumComponents(self.quad, cnc) )

    def getOrder(self):
        cdef PetscInt corder = 0
        CHKERR( PetscQuadratureGetOrder(self.quad, &corder))
        return toInt(corder)

    def setOrder(self, order):
        cdef PetscInt corder = asInt(order)
        CHKERR( PetscQuadratureSetOrder(self.quad, corder))


# --------------------------------------------------------------------
