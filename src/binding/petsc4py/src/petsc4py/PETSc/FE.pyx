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

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscFEView(self.fe, vwr) )

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
        cdef const char *cprefix = NULL
        if prefix:
             prefix = str2bytes(prefix, &cprefix)
        CHKERR( PetscFECreateDefault(ccomm, cdim, cnc, cisSimplex, cprefix, cqorder, &newfe))
        PetscCLEAR(self.obj); self.fe = newfe
        return self

    def createLagrange(self, dim, nc, isSimplex, k, qorder, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscFE newfe = NULL
        cdef PetscInt cdim = asInt(dim)
        cdef PetscInt cnc = asInt(nc)
        cdef PetscInt ck = asInt(k)
        cdef PetscInt cqorder = asInt(qorder)
        cdef PetscBool cisSimplex = asBool(isSimplex)
        CHKERR( PetscFECreateLagrange(ccomm, cdim, cnc, cisSimplex, ck, cqorder, &newfe))
        PetscCLEAR(self.obj); self.fe = newfe
        return self

    def getQuadrature(self):
        cdef Quad quad = Quad()
        CHKERR( PetscFEGetQuadrature(self.fe, &quad.quad) )
        return quad

    def getDimension(self):
        cdef PetscInt cdim = 0
        CHKERR( PetscFEGetDimension(self.fe, &cdim) )
        return toInt(cdim)

    def getSpatialDimension(self):
        cdef PetscInt csdim = 0
        CHKERR( PetscFEGetSpatialDimension(self.fe, &csdim) )
        return toInt(csdim)

    def getNumComponents(self):
        cdef PetscInt comp = 0
        CHKERR( PetscFEGetNumComponents(self.fe, &comp) )
        return toInt(comp)

    def setNumComponents(self, comp):
        cdef PetscInt ccomp = asInt(comp)
        CHKERR( PetscFESetNumComponents(self.fe, comp) )

    def getNumDof(self):
        cdef const PetscInt *numDof = NULL
        cdef PetscInt cdim = 0
        CHKERR( PetscFEGetDimension(self.fe, &cdim) )
        CHKERR( PetscFEGetNumDof(self.fe, &numDof) )
        return array_i(cdim, numDof)

    def getTileSizes(self):
        cdef PetscInt blockSize = 0, numBlocks = 0
        cdef PetscInt batchSize = 0, numBatches = 0
        CHKERR( PetscFEGetTileSizes(self.fe, &blockSize, &numBlocks, &batchSize, &numBatches) )
        return toInt(blockSize), toInt(numBlocks), toInt(batchSize), toInt(numBatches)
    
    def setTileSizes(self, blockSize, numBlocks, batchSize, numBatches):
        cdef PetscInt cblockSize = asInt(blockSize), cnumBlocks = asInt(numBlocks)
        cdef PetscInt cbatchSize = asInt(batchSize), cnumBatches = asInt(numBatches)
        CHKERR( PetscFESetTileSizes(self.fe, blockSize, numBlocks, batchSize, numBatches) )

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

    def setType(self, fe_type):
        cdef PetscFEType cval = NULL
        fe_type = str2bytes(fe_type, &cval)
        CHKERR( PetscFESetType(self.fe, cval) )
        return self

    def getBasisSpace(self):
        cdef Space sp = Space()
        CHKERR( PetscFEGetBasisSpace(self.fe, &sp.space ) )
        return sp
    
    def setBasisSpace(self, Space sp):
        CHKERR( PetscFESetBasisSpace(self.fe, sp.space ) )

    def setFromOptions(self):
        CHKERR( PetscFESetFromOptions(self.fe) )

    def setUp(self):
        CHKERR( PetscFESetUp(self.fe) )

    def getDualSpace(self):
        cdef DualSpace dspace = DualSpace()
        CHKERR( PetscFEGetDualSpace(self.fe, &dspace.dualspace) )
        return dspace
    
    def setDualSpace(self, DualSpace dspace):
        CHKERR( PetscFESetDualSpace(self.fe, dspace.dualspace) )

    def viewFromOptions(self, name, Object obj=None):
        cdef const char *cname = NULL
        _ = str2bytes(name, &cname)
        cdef PetscObject  cobj = NULL
        if obj is not None: cobj = obj.obj[0]
        CHKERR( PetscFEViewFromOptions(self.fe, cobj, cname) )

# --------------------------------------------------------------------

del FEType

# --------------------------------------------------------------------
