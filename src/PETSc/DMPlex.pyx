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

    def createBoxMesh(self, dim, interpolate=True, comm=None):
        cdef PetscInt  cdim = asInt(dim)
        cdef PetscBool interp = interpolate
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM   newdm = NULL
        CHKERR( DMPlexCreateBoxMesh(ccomm,cdim,interp,&newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createCGNS(self, cgid, interpolate=True, comm=None):
        cdef DMPlex    dm = <DMPlex>type(self)()
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef PetscInt  ccgid = asInt(cgid)
        CHKERR( DMPlexCreateCGNS(ccomm, ccgid, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createCGNSFromFile(self, filename, interpolate=True, comm=None):
        cdef DMPlex    dm = <DMPlex>type(self)()
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef const_char *cfile = NULL
        filename = str2bytes(filename, &cfile)
        CHKERR( DMPlexCreateCGNSFromFile(ccomm, cfile, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createExodusFromFile(self, filename, interpolate=True, comm=None):
        cdef DMPlex    dm = <DMPlex>type(self)()
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef const_char *cfile = NULL
        filename = str2bytes(filename, &cfile)
        CHKERR( DMPlexCreateExodusFromFile(ccomm, cfile, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createExodus(self, exoid, interpolate=True, comm=None):
        cdef DMPlex    dm = <DMPlex>type(self)()
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef PetscInt  cexoid = asInt(exoid)
        CHKERR( DMPlexCreateExodus(ccomm, cexoid, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createGmsh(self, Viewer viewer, interpolate=True, comm=None):
        cdef DMPlex    dm = <DMPlex>type(self)()
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        CHKERR( DMPlexCreateGmsh(ccomm, viewer.vwr, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createCohesiveSubmesh(self, hasLagrange, value):
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

    def hasLabel(self, label):
        cdef PetscBool flag = PETSC_FALSE
        cdef const_char *cval = NULL
        label = str2bytes(label, &cval)
        CHKERR( DMPlexHasLabel(self.dm, cval, &flag) )
        return <bint> flag

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

    def removeLabel(self,name):
        cdef const_char *cname = NULL
        cdef PetscDMLabel clbl = NULL
        name = str2bytes(name, &cname)
        CHKERR( DMPlexRemoveLabel(self.dm, cname, &clbl) )

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
        try:
            return array_i(numCoveringPoints,coveringPoints)
        finally:
            CHKERR( DMPlexRestoreMeet(self.dm,numPoints,ipoints,&numCoveringPoints,&coveringPoints) )

    def getJoin(self,points):
        cdef PetscInt  numPoints = 0
        cdef PetscInt *ipoints = NULL
        cdef PetscInt  numCoveringPoints = 0
        cdef const_PetscInt *coveringPoints = NULL
        points = iarray_i(points, &numPoints, &ipoints)
        CHKERR( DMPlexGetJoin(self.dm,numPoints,ipoints,&numCoveringPoints,&coveringPoints) )
        try:
            return array_i(numCoveringPoints,coveringPoints)
        finally:
            CHKERR( DMPlexRestoreJoin(self.dm,numPoints,ipoints,&numCoveringPoints,&coveringPoints) )

    def getTransitiveClosure(self,p,useCone=True):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscBool cuseCone = PETSC_FALSE
        if useCone: cuseCone = PETSC_TRUE
        cdef PetscInt  numPoints = 0
        cdef PetscInt *points = NULL
        CHKERR( DMPlexGetTransitiveClosure(self.dm,cp,cuseCone,&numPoints,&points) )
        try:
            out = array_i(2*numPoints,points)
        finally:
            CHKERR( DMPlexRestoreTransitiveClosure(self.dm,cp,cuseCone,&numPoints,&points) )
        return out[::2],out[1::2]

    def vecGetClosure(self, Section sec, Vec vec, p):
        cdef PetscInt cp = asInt(p), csize = 0
        cdef PetscScalar *cvals = NULL
        CHKERR( DMPlexVecGetClosure(self.dm, sec.sec, vec.vec, cp, &csize, &cvals) )
        try:
            closure = array_s(csize, cvals)
        finally:
            CHKERR( DMPlexVecRestoreClosure(self.dm, sec.sec, vec.vec, cp, &csize, &cvals) )
        return closure

    def generate(self, DMPlex boundary, name=None, interpolate=True):
        cdef DMPlex    dm = <DMPlex>type(self)()
        cdef PetscBool interp = PETSC_FALSE
        if interpolate: interp = PETSC_TRUE
        cdef const_char *cname = NULL
        if name: name = str2bytes(name, &cname)
        cdef PetscDM   newdm = NULL
        CHKERR( DMPlexGenerate(boundary.dm, cname, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createSquareBoundary(self, lower, upper, edges):
        cdef DMPlex    dm = <DMPlex>type(self)()
        cdef PetscInt nlow = 0, nup = 0, nedg = 0
        cdef PetscInt *iedg = NULL
        cdef PetscReal *ilow = NULL, *iup = NULL
        lower = iarray_r(lower, &nlow, &ilow)
        upper = iarray_r(upper, &nup,  &iup)
        edges = iarray_i(edges, &nedg, &iedg)
        CHKERR( DMPlexCreateSquareBoundary(self.dm, ilow, iup, iedg) )
        return self

    def createCubeBoundary(self, lower, upper, faces):
        cdef DMPlex    dm = <DMPlex>type(self)()
        cdef PetscInt nlow = 0, nup = 0, nfac = 0
        cdef PetscInt *ifac = NULL
        cdef PetscReal *ilow = NULL, *iup = NULL
        lower = iarray_r(lower, &nlow, &ilow)
        upper = iarray_r(upper, &nup,  &iup)
        edges = iarray_i(faces, &nfac, &ifac)
        CHKERR( DMPlexCreateCubeBoundary(self.dm, ilow, iup, ifac) )
        return self

    def markBoundaryFaces(self, label):
        if not self.hasLabel(label):
            self.createLabel(label)
        cdef const_char *cval = NULL
        label = str2bytes(label, &cval)
        cdef PetscDMLabel clbl = NULL
        CHKERR( DMPlexGetLabel(self.dm, cval, &clbl) )
        CHKERR( DMPlexMarkBoundaryFaces(self.dm, clbl) )

    def setAdjacencyUseCone(self, useCone=True):
        cdef PetscBool flag = PETSC_FALSE
        if useCone: flag = PETSC_TRUE
        CHKERR( DMPlexSetAdjacencyUseCone(self.dm, flag) )

    def setAdjacencyUseClosure(self, useClosure=True):
        cdef PetscBool flag = PETSC_FALSE
        if useClosure: flag = PETSC_TRUE
        CHKERR( DMPlexSetAdjacencyUseClosure(self.dm, flag) )

    def distribute(self, partitioner=None, overlap=0):
        cdef PetscDM pardm = NULL
        cdef PetscSF sf = NULL
        cdef const_char *cpart = NULL
        if partitioner: partitioner = str2bytes(partitioner, &cpart)
        cdef PetscInt coverlap = asInt(overlap)
        cdef SF pointsf = SF()
        CHKERR( DMPlexDistribute(self.dm, cpart, coverlap, &pointsf.sf, &pardm) )
        PetscCLEAR(self.obj); self.dm = pardm
        return pointsf

    def createSection(self, numFields, numComp, numDof, numBC=0, bcField=None, bcPoints=None, IS perm=None):
        cdef PetscInt dim = 0
        CHKERR( DMPlexGetDimension(self.dm, &dim) )
        cdef PetscInt nfield = asInt(numFields)
        cdef PetscInt ncomp = 0, ndof = 0
        cdef PetscInt *icomp = NULL, *idof = NULL
        numComp = iarray_i(numComp, &ncomp, &icomp)
        numDof = iarray_i(numDof, &ndof, &idof)

        cdef PetscInt nbcfield = 0
        cdef PetscInt *ibcfield = NULL
        cdef PetscInt nbc = asInt(numBC)
        cdef PetscIS* cbcpoints = NULL

        if numBC != 0:
          assert numBC > 0
          assert numBC == len(bcField)
          assert numBC == len(bcPoints)
          bcField = iarray_i(bcField, &nbcfield, &ibcfield)
          tmp = oarray_p(empty_p(nbc), NULL, <void**>&cbcpoints)
          for i from 0 <= i < nbc: cbcpoints[i] = (<IS?>bcPoints[i]).iset
        else:
          assert bcField is None
          assert bcPoints is None

        cdef PetscIS cperm = NULL
        if perm is not None:
            cperm = perm.iset
        cdef Section sec = Section()
        CHKERR( DMPlexCreateSection(self.dm, dim, nfield, icomp, idof, nbc, ibcfield, cbcpoints, cperm, &sec.sec) )
        return sec

    def setRefinementUniform(self, refinementUniform=True):
        cdef PetscBool uniform = refinementUniform
        CHKERR( DMPlexSetRefinementUniform(self.dm, uniform) )

    def getRefinementUniform(self):
        cdef PetscBool uniform
        CHKERR( DMPlexGetRefinementUniform(self.dm, &uniform) )
        return <bint>uniform

    def setRefinementLimit(self, refinementLimit):
        cdef PetscReal limit = asReal(refinementLimit)
        CHKERR( DMPlexSetRefinementLimit(self.dm, limit) )

    def getRefinementLimit(self):
        cdef PetscReal limit
        CHKERR( DMPlexGetRefinementLimit(self.dm, &limit) )
        return toReal(limit)

    def getOrdering(self, otype):
        cdef PetscMatOrderingType cval = NULL
        otype = str2bytes(otype, &cval)
        cdef IS perm = IS()
        CHKERR( DMPlexGetOrdering(self.dm, cval, &perm.iset) )
        return perm

    def permute(self, IS perm not None):
        cdef DMPlex dm = <DMPlex>type(self)()
        cdef PetscDM newdm = NULL

        CHKERR( DMPlexPermute(self.dm, perm.iset, &newdm) )
        dm.dm = newdm
        return dm
