# --------------------------------------------------------------------

cdef class DMPlex(DM):

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR( DMPlexCreate(ccomm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createFromCellList(self, dim, cells, coords, interpolate=True, comm=None):
        cdef DMPlex    dm = <DMPlex>type(self)()
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef PetscInt  cdim = asInt(dim)
        cdef PetscInt  numCells = 0
        cdef PetscInt  numCorners = 0
        cdef int       *cellVertices = NULL
        cdef PetscInt  numVertices = 0
        cdef PetscInt  spaceDim= 0
        cdef double    *vertexCoords = NULL
        cdef int npy_flags = NPY_ARRAY_ALIGNED|NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_CARRAY
        cells  = PyArray_FROM_OTF(cells,  NPY_INT,    npy_flags)
        coords = PyArray_FROM_OTF(coords, NPY_DOUBLE, npy_flags)
        if PyArray_NDIM(cells) != 2: raise ValueError(
                ("cell indices must have two dimensions: "
                 "cells.ndim=%d") % (PyArray_NDIM(cells)) )
        if PyArray_NDIM(coords) != 2: raise ValueError(
                ("coords vertices must have two dimensions: "
                 "coords.ndim=%d") % (PyArray_NDIM(coords)) )
        numCells     = <PetscInt> PyArray_DIM(cells,  0)
        numCorners   = <PetscInt> PyArray_DIM(cells,  1)
        numVertices  = <PetscInt> PyArray_DIM(coords, 0)
        spaceDim     = <PetscInt> PyArray_DIM(coords, 1)
        cellVertices = <int*>     PyArray_DATA(cells)
        vertexCoords = <double*>  PyArray_DATA(coords)
        CHKERR( DMPlexCreateFromCellList(ccomm,cdim,numCells,numVertices,numCorners,interp,cellVertices,spaceDim,vertexCoords,&newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createCohesiveSubmesh(self,hasLagrange,value):
        cdef PetscBool hasL = hasLagrange
        cdef PetscInt cvalue = asInt(value)
        cdef DM subdm = DMPlex()
        CHKERR( DMPlexCreateCohesiveSubmesh(self.dm,hasL,NULL,cvalue,&subdm.dm) )
        return subdm
    
    def getDimension(self):
        cdef PetscInt dim = 0
        CHKERR( DMPlexGetDimension(self.dm, &dim) )
        return toInt(dim)

    def setDimension(self, dim):
        cdef PetscInt cdim = asInt(dim)
        CHKERR( DMPlexSetDimension(self.dm, cdim) )

    def getChart(self):
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        return toInt(pStart), toInt(pEnd)

    def setChart(self, pStart, pEnd):
        cdef PetscInt cStart = asInt(pStart)
        cdef PetscInt cEnd   = asInt(pEnd)
        CHKERR( DMPlexSetChart(self.dm, cStart, cEnd) )

    def getConeSize(self, p):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscInt csize = 0
        CHKERR( DMPlexGetConeSize(self.dm, cp, &csize) )
        return toInt(csize)

    def setConeSize(self, p, size):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscInt csize = asInt(size)
        CHKERR( DMPlexSetConeSize(self.dm, cp, csize) )

    def getCone(self, p):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscInt        ncone = 0
        cdef const_PetscInt *icone = NULL
        CHKERR( DMPlexGetConeSize(self.dm, cp, &ncone) )
        CHKERR( DMPlexGetCone(self.dm, cp, &icone) )
        return array_i(ncone, icone)

    def setCone(self, p, cone, orientation=None):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        #
        cdef PetscInt  ncone = 0
        cdef PetscInt *icone = NULL
        cone = iarray_i(cone, &ncone, &icone)
        CHKERR( DMPlexSetConeSize(self.dm, cp, ncone) )
        CHKERR( DMPlexSetCone(self.dm, cp, icone) )
        #
        cdef PetscInt  norie = 0
        cdef PetscInt *iorie = NULL
        if orientation is not None:
            orientation = iarray_i(orientation, &norie, &iorie)
            assert norie == ncone 
            CHKERR( DMPlexSetConeOrientation(self.dm, cp, iorie) )

    def insertCone(self, p, conePos, conePoint):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt cconePos = asInt(conePos)
        cdef PetscInt cconePoint = asInt(conePoint)
        CHKERR( DMPlexInsertCone(self.dm,cp,cconePos,cconePoint) )

    def insertConeOrientation(self, p, conePos, coneOrientation):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt cconePos = asInt(conePos)
        cdef PetscInt cconeOrientation = asInt(coneOrientation)
        CHKERR( DMPlexInsertConeOrientation(self.dm,cp,cconePos,cconeOrientation) )

    def getConeOrientation(self, p):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscInt        norie = 0
        cdef const_PetscInt *iorie = NULL
        CHKERR( DMPlexGetConeSize(self.dm, cp, &norie) )
        CHKERR( DMPlexGetConeOrientation(self.dm, cp, &iorie) )
        return array_i(norie, iorie)

    def setConeOrientation(self, p, orientation):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscInt ncone = 0
        CHKERR( DMPlexGetConeSize(self.dm, cp, &ncone) )
        cdef PetscInt  norie = 0
        cdef PetscInt *iorie = NULL
        orientation = iarray_i(orientation, &norie, &iorie)
        assert norie == ncone 
        CHKERR( DMPlexSetConeOrientation(self.dm, cp, iorie) )
        
    def getSupportSize(self, p):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscInt ssize = 0
        CHKERR( DMPlexGetSupportSize(self.dm, cp, &ssize) )
        return toInt(ssize)

    def setSupportSize(self, p, size):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscInt ssize = asInt(size)
        CHKERR( DMPlexSetSupportSize(self.dm, cp, ssize) )

    def getSupport(self, p):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscInt        nsupp = 0
        cdef const_PetscInt *isupp = NULL
        CHKERR( DMPlexGetSupportSize(self.dm, cp, &nsupp) )
        CHKERR( DMPlexGetSupport(self.dm, cp, &isupp) )
        return array_i(nsupp, isupp)

    def setSupport(self, p, supp):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscInt  nsupp = 0
        cdef PetscInt *isupp = NULL
        supp = iarray_i(supp, &nsupp, &isupp)
        CHKERR( DMPlexSetSupportSize(self.dm, cp, nsupp) )
        CHKERR( DMPlexSetSupport(self.dm, cp, isupp) )

    def getMaxSizes(self):
        cdef PetscInt maxConeSize = 0, maxSupportSize = 0
        CHKERR( DMPlexGetMaxSizes(self.dm, &maxConeSize, &maxSupportSize) )
        return toInt(maxConeSize), toInt(maxSupportSize)

    def symmetrize(self):
        CHKERR( DMPlexSymmetrize(self.dm) )

    def stratify(self):
        CHKERR( DMPlexStratify(self.dm) )
        
    def orient(self):
        CHKERR( DMPlexOrient(self.dm) )

    def getNumLabels(self):
        cdef PetscInt nLabels = 0
        CHKERR( DMPlexGetNumLabels(self.dm,&nLabels) )
        return toInt(nLabels)

    def getLabelName(self, n):
        cdef PetscInt cn = asInt(n)
        cdef const_char *cname = NULL
        CHKERR( DMPlexGetLabelName(self.dm,cn,&cname) )
        return bytes2str(cname)

    def getCellNumbering(self):
        cdef IS globalCellNumbers = IS()
        CHKERR( DMPlexGetCellNumbering(self.dm,&globalCellNumbers.iset) )
        return globalCellNumbers

    def getVertexNumbering(self):
        cdef IS globalVertexNumbers = IS()
        CHKERR( DMPlexGetVertexNumbering(self.dm,&globalVertexNumbers.iset) )
        return globalVertexNumbers

    def createLabel(self,name):
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMPlexCreateLabel(self.dm,cname) )

    def getLabelValue(self, name, n):
        cdef PetscInt cn = asInt(n), value
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMPlexGetLabelValue(self.dm,cname,cn,&value) )
        return toInt(value)
    
    def setLabelValue(self, name, n, value):
        cdef PetscInt cn = asInt(n), cvalue = asInt(value)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMPlexSetLabelValue(self.dm,cname,cn,cvalue) )

    def clearLabelValue(self, name, n, value):
        cdef PetscInt cn = asInt(n), cvalue = asInt(value)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMPlexClearLabelValue(self.dm,cname,cn,cvalue) )

    def getLabelSize(self, name):
        cdef PetscInt size = 0
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMPlexGetLabelSize(self.dm,cname,&size) )
        return toInt(size)
    
    def getLabelIdIS(self, name):
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        cdef IS lis = IS()
        CHKERR( DMPlexGetLabelIdIS(self.dm,cname,&lis.iset) )
        return lis
    
    def getStratumSize(self, name, n):
        cdef PetscInt size = 0
        cdef PetscInt cn = asInt(n)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMPlexGetStratumSize(self.dm,cname,n,&size) )
        return toInt(size)
    
    def getStratumIS(self, name, n):
        cdef PetscInt cn = asInt(n)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        cdef IS sis = IS()
        CHKERR( DMPlexGetStratumIS(self.dm,cname,n,&sis.iset) )
        return sis
    
    def clearLabelStratum(self, name, n):
        cdef PetscInt cn = asInt(n)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMPlexClearLabelStratum(self.dm,cname,n) )

    def getDepth(self):
        cdef PetscInt depth = 0
        CHKERR( DMPlexGetDepth(self.dm,&depth) )
        return toInt(depth)
    
    def getDepthStratum(self,svalue):
        cdef PetscInt csvalue = asInt(svalue),sStart,sEnd
        CHKERR( DMPlexGetDepthStratum(self.dm,csvalue,&sStart,&sEnd) )
        return (toInt(sStart),toInt(sEnd))
    
    def getHeightStratum(self,svalue):
        cdef PetscInt csvalue = asInt(svalue),sStart,sEnd
        CHKERR( DMPlexGetHeightStratum(self.dm,csvalue,&sStart,&sEnd) )
        return (toInt(sStart),toInt(sEnd))

    def getMeet(self,points):
        cdef PetscInt  numPoints = 0
        cdef PetscInt *ipoints = NULL
        cdef PetscInt  numCoveringPoints = 0
        cdef const_PetscInt *coveringPoints = NULL
        points = iarray_i(points, &numPoints, &ipoints)
        CHKERR( DMPlexGetMeet(self.dm,numPoints,ipoints,&numCoveringPoints,&coveringPoints) )
        return array_i(numCoveringPoints,coveringPoints)

    def getJoin(self,points):
        cdef PetscInt  numPoints = 0
        cdef PetscInt *ipoints = NULL
        cdef PetscInt  numCoveringPoints = 0
        cdef const_PetscInt *coveringPoints = NULL
        points = iarray_i(points, &numPoints, &ipoints)
        CHKERR( DMPlexGetJoin(self.dm,numPoints,ipoints,&numCoveringPoints,&coveringPoints) )
        return array_i(numCoveringPoints,coveringPoints)

    def getTransitiveClosure(self,p,useCone=True):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscBool cuseCone = useCone
        cdef PetscInt  numPoints = 0
        cdef PetscInt *points = NULL
        CHKERR( DMPlexGetTransitiveClosure(self.dm,cp,cuseCone,&numPoints,&points) )
        out = array_i(2*numPoints,points)
        return out[::2],out[1::2]
