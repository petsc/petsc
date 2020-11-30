# --------------------------------------------------------------------

cdef class DMPlex(DM):

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR( DMPlexCreate(ccomm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createFromCellList(self, dim, cells, coords, interpolate=True, comm=None):
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef PetscInt  cdim = asInt(dim)
        cdef PetscInt  numCells = 0
        cdef PetscInt  numCorners = 0
        cdef PetscInt  *cellVertices = NULL
        cdef PetscInt  numVertices = 0
        cdef PetscInt  spaceDim= 0
        cdef PetscReal *vertexCoords = NULL
        cdef int npy_flags = NPY_ARRAY_ALIGNED|NPY_ARRAY_NOTSWAPPED|NPY_ARRAY_CARRAY
        cells  = PyArray_FROM_OTF(cells,  NPY_PETSC_INT,  npy_flags)
        coords = PyArray_FROM_OTF(coords, NPY_PETSC_REAL, npy_flags)
        if PyArray_NDIM(cells) != 2: raise ValueError(
                ("cell indices must have two dimensions: "
                 "cells.ndim=%d") % (PyArray_NDIM(cells)) )
        if PyArray_NDIM(coords) != 2: raise ValueError(
                ("coords vertices must have two dimensions: "
                 "coords.ndim=%d") % (PyArray_NDIM(coords)) )
        numCells     = <PetscInt>   PyArray_DIM(cells,  0)
        numCorners   = <PetscInt>   PyArray_DIM(cells,  1)
        numVertices  = <PetscInt>   PyArray_DIM(coords, 0)
        spaceDim     = <PetscInt>   PyArray_DIM(coords, 1)
        cellVertices = <PetscInt*>  PyArray_DATA(cells)
        vertexCoords = <PetscReal*> PyArray_DATA(coords)
        CHKERR( DMPlexCreateFromCellListPetsc(ccomm, cdim, numCells, numVertices,
                                              numCorners, interp, cellVertices,
                                              spaceDim, vertexCoords, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createBoxMesh(self, faces, lower=(0,0,0), upper=(1,1,1),
                      simplex=True, periodic=False, interpolate=True, comm=None):
        cdef Py_ssize_t i = 0
        cdef PetscInt dim = 0, *cfaces = NULL
        faces = iarray_i(faces, &dim, &cfaces)
        assert dim >= 1 and dim <= 3
        cdef PetscReal clower[3]
        clower[0] = clower[1] = clower[2] = 0
        for i from 0 <= i < dim: clower[i] = lower[i]
        cdef PetscReal cupper[3]
        cupper[0] = cupper[1] = cupper[2] = 1
        for i from 0 <= i < dim: cupper[i] = upper[i]
        cdef PetscDMBoundaryType btype[3];
        asBoundary(periodic, &btype[0], &btype[1], &btype[2])
        cdef PetscBool csimplex = simplex
        cdef PetscBool cinterp = interpolate
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM   newdm = NULL
        CHKERR( DMPlexCreateBoxMesh(ccomm, dim, csimplex, cfaces,
                                    clower, cupper, btype, cinterp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createFromFile(self, filename, interpolate=True, comm=None):
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef const char *cfile = NULL
        filename = str2bytes(filename, &cfile)
        CHKERR( DMPlexCreateFromFile(ccomm, cfile, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createCGNS(self, cgid, interpolate=True, comm=None):
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef PetscInt  ccgid = asInt(cgid)
        CHKERR( DMPlexCreateCGNS(ccomm, ccgid, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createCGNSFromFile(self, filename, interpolate=True, comm=None):
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef const char *cfile = NULL
        filename = str2bytes(filename, &cfile)
        CHKERR( DMPlexCreateCGNSFromFile(ccomm, cfile, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createExodusFromFile(self, filename, interpolate=True, comm=None):
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef const char *cfile = NULL
        filename = str2bytes(filename, &cfile)
        CHKERR( DMPlexCreateExodusFromFile(ccomm, cfile, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createExodus(self, exoid, interpolate=True, comm=None):
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        cdef PetscInt  cexoid = asInt(exoid)
        CHKERR( DMPlexCreateExodus(ccomm, cexoid, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createGmsh(self, Viewer viewer, interpolate=True, comm=None):
        cdef MPI_Comm  ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool interp = interpolate
        cdef PetscDM   newdm = NULL
        CHKERR( DMPlexCreateGmsh(ccomm, viewer.vwr, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def createCohesiveSubmesh(self, hasLagrange, value):
        cdef PetscBool flag = hasLagrange
        cdef PetscInt cvalue = asInt(value)
        cdef DM subdm = DMPlex()
        CHKERR( DMPlexCreateCohesiveSubmesh(self.dm, flag, NULL, cvalue, &subdm.dm) )
        return subdm

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
        cdef const PetscInt *icone = NULL
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
        CHKERR( DMPlexInsertConeOrientation(self.dm, cp, cconePos, cconeOrientation) )

    def getConeOrientation(self, p):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscInt        norie = 0
        cdef const PetscInt *iorie = NULL
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
        cdef const PetscInt *isupp = NULL
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

    def getCellNumbering(self):
        cdef IS iset = IS()
        CHKERR( DMPlexGetCellNumbering(self.dm, &iset.iset) )
        PetscINCREF(iset.obj)
        return iset

    def getVertexNumbering(self):
        cdef IS iset = IS()
        CHKERR( DMPlexGetVertexNumbering(self.dm, &iset.iset) )
        PetscINCREF(iset.obj)
        return iset

    def createPointNumbering(self):
        cdef IS iset = IS()
        CHKERR( DMPlexCreatePointNumbering(self.dm, &iset.iset) )
        return iset

    def getDepth(self):
        cdef PetscInt depth = 0
        CHKERR( DMPlexGetDepth(self.dm,&depth) )
        return toInt(depth)

    def getDepthStratum(self, svalue):
        cdef PetscInt csvalue = asInt(svalue), sStart = 0, sEnd = 0
        CHKERR( DMPlexGetDepthStratum(self.dm, csvalue, &sStart, &sEnd) )
        return (toInt(sStart), toInt(sEnd))

    def getHeightStratum(self, svalue):
        cdef PetscInt csvalue = asInt(svalue), sStart = 0, sEnd = 0
        CHKERR( DMPlexGetHeightStratum(self.dm, csvalue, &sStart, &sEnd) )
        return (toInt(sStart), toInt(sEnd))

    def getMeet(self, points):
        cdef PetscInt  numPoints = 0
        cdef PetscInt *ipoints = NULL
        cdef PetscInt  numCoveringPoints = 0
        cdef const PetscInt *coveringPoints = NULL
        points = iarray_i(points, &numPoints, &ipoints)
        CHKERR( DMPlexGetMeet(self.dm, numPoints, ipoints, &numCoveringPoints, &coveringPoints) )
        try:
            return array_i(numCoveringPoints, coveringPoints)
        finally:
            CHKERR( DMPlexRestoreMeet(self.dm, numPoints, ipoints, &numCoveringPoints, &coveringPoints) )

    def getJoin(self, points):
        cdef PetscInt  numPoints = 0
        cdef PetscInt *ipoints = NULL
        cdef PetscInt  numCoveringPoints = 0
        cdef const PetscInt *coveringPoints = NULL
        points = iarray_i(points, &numPoints, &ipoints)
        CHKERR( DMPlexGetJoin(self.dm, numPoints, ipoints, &numCoveringPoints, &coveringPoints) )
        try:
            return array_i(numCoveringPoints, coveringPoints)
        finally:
            CHKERR( DMPlexRestoreJoin(self.dm, numPoints, ipoints, &numCoveringPoints, &coveringPoints) )

    def getTransitiveClosure(self, p, useCone=True):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt pStart = 0, pEnd = 0
        CHKERR( DMPlexGetChart(self.dm, &pStart, &pEnd) )
        assert cp>=pStart and cp<pEnd
        cdef PetscBool cuseCone = useCone
        cdef PetscInt  numPoints = 0
        cdef PetscInt *points = NULL
        CHKERR( DMPlexGetTransitiveClosure(self.dm, cp, cuseCone, &numPoints, &points) )
        try:
            out = array_i(2*numPoints,points)
        finally:
            CHKERR( DMPlexRestoreTransitiveClosure(self.dm, cp, cuseCone, &numPoints, &points) )
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

    def getVecClosure(self, Section sec or None, Vec vec, point):
        cdef PetscSection csec = sec.sec if sec is not None else NULL
        cdef PetscInt cp = asInt(point), csize = 0
        cdef PetscScalar *cvals = NULL
        CHKERR( DMPlexVecGetClosure(self.dm, csec, vec.vec, cp, &csize, &cvals) )
        try:
            closure = array_s(csize, cvals)
        finally:
            CHKERR( DMPlexVecRestoreClosure(self.dm, csec, vec.vec, cp, &csize, &cvals) )
        return closure

    def setVecClosure(self, Section sec or None, Vec vec, point, values, addv=None):
        cdef PetscSection csec = sec.sec if sec is not None else NULL
        cdef PetscInt cp = asInt(point)
        cdef PetscInt csize = 0
        cdef PetscScalar *cvals = NULL
        cdef object tmp = iarray_s(values, &csize, &cvals)
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMPlexVecSetClosure(self.dm, csec, vec.vec, cp, cvals, im) )

    def setMatClosure(self, Section sec or None, Section gsec or None,
                      Mat mat, point, values, addv=None):
        cdef PetscSection csec  =  sec.sec if  sec is not None else NULL
        cdef PetscSection cgsec = gsec.sec if gsec is not None else NULL
        cdef PetscInt cp = asInt(point)
        cdef PetscInt csize = 0
        cdef PetscScalar *cvals = NULL
        cdef object tmp = iarray_s(values, &csize, &cvals)
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMPlexMatSetClosure(self.dm, csec, cgsec, mat.mat, cp, cvals, im) )

    def generate(self, DMPlex boundary, name=None, interpolate=True):
        cdef PetscBool interp = interpolate
        cdef const char *cname = NULL
        if name: name = str2bytes(name, &cname)
        cdef PetscDM   newdm = NULL
        CHKERR( DMPlexGenerate(boundary.dm, cname, interp, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def setTriangleOptions(self, opts):
        cdef const char *copts = NULL
        opts = str2bytes(opts, &copts)
        CHKERR( DMPlexTriangleSetOptions(self.dm, copts) )

    def setTetGenOptions(self, opts):
        cdef const char *copts = NULL
        opts = str2bytes(opts, &copts)
        CHKERR( DMPlexTetgenSetOptions(self.dm, copts) )

    def createSquareBoundary(self, lower, upper, edges):
        cdef PetscInt nlow = 0, nup = 0, nedg = 0
        cdef PetscInt *iedg = NULL
        cdef PetscReal *ilow = NULL, *iup = NULL
        lower = iarray_r(lower, &nlow, &ilow)
        upper = iarray_r(upper, &nup,  &iup)
        edges = iarray_i(edges, &nedg, &iedg)
        CHKERR( DMPlexCreateSquareBoundary(self.dm, ilow, iup, iedg) )
        return self

    def createCubeBoundary(self, lower, upper, faces):
        cdef PetscInt nlow = 0, nup = 0, nfac = 0
        cdef PetscInt *ifac = NULL
        cdef PetscReal *ilow = NULL, *iup = NULL
        lower = iarray_r(lower, &nlow, &ilow)
        upper = iarray_r(upper, &nup,  &iup)
        faces = iarray_i(faces, &nfac, &ifac)
        CHKERR( DMPlexCreateCubeBoundary(self.dm, ilow, iup, ifac) )
        return self

    def markBoundaryFaces(self, label, value=None):
        cdef PetscInt ival = PETSC_DETERMINE
        if value is not None: ival = asInt(value)
        if not self.hasLabel(label):
            self.createLabel(label)
        cdef const char *cval = NULL
        label = str2bytes(label, &cval)
        cdef PetscDMLabel clbl = NULL
        CHKERR( DMGetLabel(self.dm, cval, &clbl) )
        CHKERR( DMPlexMarkBoundaryFaces(self.dm, ival, clbl) )

    def setAdjacencyUseAnchors(self, useAnchors=True):
        cdef PetscBool flag = useAnchors
        CHKERR( DMPlexSetAdjacencyUseAnchors(self.dm, flag) )

    def getAdjacencyUseAnchors(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( DMPlexGetAdjacencyUseAnchors(self.dm, &flag) )
        return toBool(flag)

    def getAdjacency(self, p):
        cdef PetscInt cp = asInt(p)
        cdef PetscInt nadj = PETSC_DETERMINE
        cdef PetscInt *iadj = NULL
        CHKERR( DMPlexGetAdjacency(self.dm, cp, &nadj, &iadj) )
        try:
            adjacency = array_i(nadj, iadj)
        finally:
            CHKERR( PetscFree(iadj) )
        return adjacency

    def setPartitioner(self, Partitioner part):
        CHKERR( DMPlexSetPartitioner(self.dm, part.part) )

    def getPartitioner(self):
        cdef Partitioner part = Partitioner()
        CHKERR( DMPlexGetPartitioner(self.dm, &part.part) )
        PetscINCREF(part.obj)
        return part

    def rebalanceSharedPoints(self, entityDepth=0, useInitialGuess=True, parallel=True):
        cdef PetscInt centityDepth = asInt(entityDepth)
        cdef PetscBool cuseInitialGuess = asBool(useInitialGuess)
        cdef PetscBool cparallel = asBool(parallel)
        cdef PetscBool csuccess = PETSC_FALSE
        CHKERR( DMPlexRebalanceSharedPoints(self.dm, centityDepth, cuseInitialGuess, cparallel, &csuccess) )
        return toBool(csuccess)

    def distribute(self, overlap=0):
        cdef PetscDM dmParallel = NULL
        cdef PetscInt coverlap = asInt(overlap)
        cdef SF sf = SF()
        CHKERR( DMPlexDistribute(self.dm, coverlap, &sf.sf, &dmParallel) )
        if dmParallel != NULL:
            PetscCLEAR(self.obj); self.dm = dmParallel
            return sf

    def distributeOverlap(self, overlap=0):
        cdef PetscInt coverlap = asInt(overlap)
        cdef SF sf = SF()
        cdef PetscDM dmOverlap = NULL
        CHKERR( DMPlexDistributeOverlap(self.dm, coverlap,
                                        &sf.sf, &dmOverlap) )
        PetscCLEAR(self.obj); self.dm = dmOverlap
        return sf

    def isDistributed(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( DMPlexIsDistributed(self.dm, &flag) )
        return toBool(flag)

    def interpolate(self):
        cdef PetscDM newdm = NULL
        CHKERR( DMPlexInterpolate(self.dm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm

    def uninterpolate(self):
        cdef PetscDM newdm = NULL
        CHKERR( DMPlexUninterpolate(self.dm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm

    def distributeField(self, SF sf, Section sec, Vec vec,
                        Section newsec=None, Vec newvec=None):
        cdef MPI_Comm ccomm = MPI_COMM_NULL
        if newsec is None: newsec = Section()
        if newvec is None: newvec = Vec()
        if newsec.sec == NULL:
            CHKERR( PetscObjectGetComm(<PetscObject>sec.sec, &ccomm) )
            CHKERR( PetscSectionCreate(ccomm, &newsec.sec) )
        if newvec.vec == NULL:
            CHKERR( PetscObjectGetComm(<PetscObject>vec.vec, &ccomm) )
            CHKERR( VecCreate(ccomm, &newvec.vec) )
        CHKERR( DMPlexDistributeField(self.dm, sf.sf,
                                      sec.sec, vec.vec,
                                      newsec.sec, newvec.vec))
        return (newsec, newvec)

    def createCoarsePointIS(self):
        cdef IS fpoint = IS()
        CHKERR( DMPlexCreateCoarsePointIS(self.dm, &fpoint.iset) )
        return fpoint

    def createSection(self, numComp, numDof,
                      bcField=None, bcComps=None, bcPoints=None,
                      IS perm=None):
        # topological dimension
        cdef PetscInt dim = 0
        CHKERR( DMGetDimension(self.dm, &dim) )
        # components and DOFs
        cdef PetscInt ncomp = 0, ndof = 0
        cdef PetscInt *icomp = NULL, *idof = NULL
        numComp = iarray_i(numComp, &ncomp, &icomp)
        numDof  = iarray_i(numDof, &ndof, &idof)
        assert ndof == ncomp*(dim+1)
        # boundary conditions
        cdef PetscInt nbc = 0, i = 0
        cdef PetscInt *bcfield = NULL
        cdef PetscIS *bccomps  = NULL
        cdef PetscIS *bcpoints = NULL
        if bcField is not None:
            bcField = iarray_i(bcField, &nbc, &bcfield)
            if bcComps is not None:
                bcComps = list(bcComps)
                assert len(bcComps) == nbc 
                tmp1 = oarray_p(empty_p(nbc), NULL, <void**>&bccomps)
                for i from 0 <= i < nbc:
                    bccomps[i] = (<IS?>bcComps[<Py_ssize_t>i]).iset
            if bcPoints is not None:
                bcPoints = list(bcPoints)
                assert len(bcPoints) == nbc
                tmp2 = oarray_p(empty_p(nbc), NULL, <void**>&bcpoints)
                for i from 0 <= i < nbc:
                    bcpoints[i] = (<IS?>bcPoints[<Py_ssize_t>i]).iset
            else:
                raise ValueError("bcPoints is a required argument")
        else:
            assert bcComps  is None
            assert bcPoints is None
        # optional chart permutations
        cdef PetscIS cperm = NULL
        if perm is not None: cperm = perm.iset
        # create section
        cdef Section sec = Section()
        CHKERR( DMPlexCreateSection(self.dm, NULL, icomp, idof,
                                    nbc, bcfield, bccomps, bcpoints,
                                    cperm, &sec.sec) )
        return sec

    def getPointLocal(self, point):
        cdef PetscInt start = 0, end = 0
        cdef PetscInt cpoint = asInt(point)
        CHKERR( DMPlexGetPointLocal(self.dm, cpoint, &start, &end) )
        return toInt(start), toInt(end)

    def getPointLocalField(self, point, field):
        cdef PetscInt start = 0, end = 0
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        CHKERR( DMPlexGetPointLocalField(self.dm, cpoint, cfield, &start, &end) )
        return toInt(start), toInt(end)

    def getPointGlobal(self, point):
        cdef PetscInt start = 0, end = 0
        cdef PetscInt cpoint = asInt(point)
        CHKERR( DMPlexGetPointGlobal(self.dm, cpoint, &start, &end) )
        return toInt(start), toInt(end)

    def getPointGlobalField(self, point, field):
        cdef PetscInt start = 0, end = 0
        cdef PetscInt cpoint = asInt(point)
        cdef PetscInt cfield = asInt(field)
        CHKERR( DMPlexGetPointGlobalField(self.dm, cpoint, cfield, &start, &end) )
        return toInt(start), toInt(end)

    def createClosureIndex(self, Section sec or None):
        cdef PetscSection csec = sec.sec if sec is not None else NULL
        CHKERR( DMPlexCreateClosureIndex(self.dm, csec) )

    #

    def setRefinementUniform(self, refinementUniform=True):
        cdef PetscBool flag = refinementUniform
        CHKERR( DMPlexSetRefinementUniform(self.dm, flag) )

    def getRefinementUniform(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( DMPlexGetRefinementUniform(self.dm, &flag) )
        return toBool(flag)

    def setRefinementLimit(self, refinementLimit):
        cdef PetscReal rval = asReal(refinementLimit)
        CHKERR( DMPlexSetRefinementLimit(self.dm, rval) )

    def getRefinementLimit(self):
        cdef PetscReal rval = 0.0
        CHKERR( DMPlexGetRefinementLimit(self.dm, &rval) )
        return toReal(rval)

    def getOrdering(self, otype):
        cdef PetscMatOrderingType cval = NULL
        cdef PetscDMLabel label = NULL
        otype = str2bytes(otype, &cval)
        cdef IS perm = IS()
        CHKERR( DMPlexGetOrdering(self.dm, cval, label, &perm.iset) )
        return perm

    def permute(self, IS perm):
        cdef DMPlex dm = <DMPlex>type(self)()
        CHKERR( DMPlexPermute(self.dm, perm.iset, &dm.dm) )
        return dm

    #

    def computeCellGeometryFVM(self, cell):
        cdef PetscInt dim = 0
        cdef PetscInt ccell = asInt(cell)
        CHKERR( DMGetDimension(self.dm, &dim) )
        cdef PetscReal vol = 0, centroid[3], normal[3]
        CHKERR( DMPlexComputeCellGeometryFVM(self.dm, ccell, &vol, centroid, normal) )
        return (toReal(vol), array_r(dim, centroid), array_r(dim, normal))

    def constructGhostCells(self, labelName=None):
        cdef const char *cname = NULL
        labelName = str2bytes(labelName, &cname)
        cdef PetscInt numGhostCells = 0
        cdef PetscDM dmGhosted = NULL
        CHKERR( DMPlexConstructGhostCells(self.dm, cname, &numGhostCells, &dmGhosted))
        PetscCLEAR(self.obj); self.dm = dmGhosted
        return toInt(numGhostCells)
