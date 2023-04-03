# --------------------------------------------------------------------

class SpaceType(object):
    POLYNOMIAL = S_(PETSCSPACEPOLYNOMIAL)
    PTRIMMED   = S_(PETSCSPACEPTRIMMED)
    TENSOR     = S_(PETSCSPACETENSOR)
    SUM        = S_(PETSCSPACESUM)
    POINT      = S_(PETSCSPACEPOINT)
    SUBSPACE   = S_(PETSCSPACESUBSPACE)
    WXY        = S_(PETSCSPACEWXY)

# --------------------------------------------------------------------

cdef class Space(Object):

    Type = SpaceType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.space
        self.space  = NULL
    
    def setUp(self):
        CHKERR( PetscSpaceSetUp(self.space) )

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSpace newsp = NULL
        CHKERR( PetscSpaceCreate(ccomm, &newsp) )
        PetscCLEAR(self.obj); self.space = newsp
        return self
    
    def destroy(self):
        CHKERR( PetscSpaceDestroy(&self.space) )
        return self
    
    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscSpaceView(self.space, vwr) )
    
    def setFromOptions(self):
        CHKERR( PetscSpaceSetFromOptions(self.space) )

    def getDimension(self):
        cdef PetscInt cdim
        CHKERR( PetscSpaceGetDimension(self.space, &cdim))
        return toInt(cdim)

    def getDegree(self):
        cdef PetscInt cdegmax, cdegmin
        CHKERR( PetscSpaceGetDegree(self.space, &cdegmin, &cdegmax))
        return toInt(cdegmin), toInt(cdegmax)

    def setDegree(self, degree, maxDegree):
        assert( (degree != None) & (maxDegree != None))
        cdef PetscInt cdegree = PETSC_DETERMINE
        if degree is not None: cdegree = asInt(degree)
        cdef PetscInt cmaxdegree = PETSC_DETERMINE
        if maxDegree is not None: cmaxdegree = asInt(maxDegree)
        CHKERR( PetscSpaceSetDegree(self.space, cdegree, cmaxdegree) )

    def getNumVariables(self):
        cdef PetscInt cnvars
        CHKERR( PetscSpaceGetNumVariables(self.space, &cnvars))
        return toInt(cnvars)
    
    def setNumVariables(self, n):
        cdef PetscInt cn = asInt(n)
        CHKERR( PetscSpaceSetNumVariables(self.space, cn) )
    
    def getNumComponents(self):
        cdef PetscInt cncomps
        CHKERR( PetscSpaceGetNumComponents(self.space, &cncomps))
        return toInt(cncomps)

    def setNumComponents(self, nc):
        cdef PetscInt cnc = asInt(nc)
        CHKERR( PetscSpaceSetNumComponents(self.space, cnc) )

    #def evaluate(self, points):
    #    cdef PetscInt  cnpoints = 0, cdim=0, cnfuncs=0
    #    cdef PetscReal *cpoints = NULL
    #    cdef PetscReal *B = NULL, *D = NULL, *H = NULL
    #    points = iarray_r(points, &cnpoints,  &cpoints)
    #    # Get the dimension of the space
    #    CHKERR( PetscSpaceGetDimension( self.space, &cnfuncs) )
    #    CHKERR( PetscSpace)
    #    CHKERR( PetscSpaceEvaluate(self.space, cnpoints, &cpoints, &B, &D, &H) )
    #    return array_r(cnpoints*cdim, B), array_r(cnpoints*cnc, D), array_r(, H)    
    
    def getType(self):
        cdef PetscSpaceType cval = NULL
        CHKERR( PetscSpaceGetType(self.space, &cval) )
        return bytes2str(cval)

    def setType(self, space_type):
        cdef PetscSpaceType cval = NULL
        space_type = str2bytes(space_type, &cval)
        CHKERR( PetscSpaceSetType(self.space, cval) )
        return self

    def getSumConcatenate(self):
        cdef PetscBool concatenate
        CHKERR( PetscSpaceSumGetConcatenate(self.space, &concatenate))
        return toBool(concatenate)

    def setSumConcatenate(self, concatenate):
        cdef PetscBool cconcatenate = asBool(concatenate)
        CHKERR( PetscSpaceSumSetConcatenate(self.space, concatenate))
    
    def getSumNumSubspaces(self):
        cdef PetscInt numSumSpaces
        CHKERR( PetscSpaceSumGetNumSubspaces(self.space, &numSumSpaces))
        return toInt(numSumSpaces)

    def getSumSubspace(self, s):
        cdef Space subsp = Space()
        cdef PetscInt cs = asInt(s)
        CHKERR( PetscSpaceSumGetSubspace(self.space, s, &subsp.space) )
        return subsp
    
    def setSumSubspace(self, s, Space subsp):
        cdef PetscInt cs = asInt(s)
        CHKERR( PetscSpaceSumSetSubspace(self.space, cs, subsp.space) )

    def setSumNumSubspaces(self, numSumSpaces):
        cdef PetscInt cnumSumSpaces = asInt(numSumSpaces)
        CHKERR( PetscSpaceSumSetNumSubspaces(self.space, cnumSumSpaces) )

    def getTensorNumSubspaces(self):
        cdef PetscInt cnumTensSpaces = 0
        CHKERR( PetscSpaceTensorGetNumSubspaces(self.space, &cnumTensSpaces) )
        return toInt(cnumTensSpaces)
    
    def setTensorSubspace(self, s, Space subsp):
        cdef PetscInt cs = asInt(s)
        CHKERR( PetscSpaceTensorSetSubspace(self.space, cs, subsp.space) )

    def getTensorSubspace(self, s):
        cdef PetscInt cs = asInt(s)
        cdef Space subsp = Space()
        CHKERR( PetscSpaceTensorGetSubspace(self.space, cs, &subsp.space) )
        return subsp

    def setTensorNumSubspaces(self, numTensSpaces):
        cdef PetscInt cnumTensSpaces = asInt(numTensSpaces)
        CHKERR( PetscSpaceTensorSetNumSubspaces(self.space, cnumTensSpaces) )

    def getPolynomialTensor(self):
        cdef PetscBool ctensor
        CHKERR( PetscSpacePolynomialGetTensor(self.space, &ctensor) )
        return toBool(ctensor)

    def setPolynomialTensor(self, tensor):
        cdef PetscBool ctensor = asBool(tensor)
        CHKERR( PetscSpacePolynomialSetTensor(self.space, ctensor) )

    def setPointPoints(self, Quad quad):
        CHKERR( PetscSpacePointSetPoints(self.space, quad.quad))

    def getPointPoints(self):
        cdef Quad quad = Quad()
        CHKERR( PetscSpacePointGetPoints(self.space, &quad.quad))
        return quad

    def setPTrimmedFormDegree(self, formDegree):
        cdef PetscInt cformDegree = asInt(formDegree)
        CHKERR( PetscSpacePTrimmedSetFormDegree(self.space, cformDegree) )

    def getPTrimmedFormDegree(self):
        cdef PetscInt cformDegree = 0
        CHKERR( PetscSpacePTrimmedGetFormDegree(self.space, &cformDegree) )
        return toInt(cformDegree)

    def viewFromOptions(self, name, Object obj=None):
        cdef const char *cname = NULL
        _ = str2bytes(name, &cname)
        cdef PetscObject  cobj = NULL
        if obj is not None: cobj = obj.obj[0]
        CHKERR( PetscSpaceViewFromOptions(self.space, cobj, cname) )
    
# --------------------------------------------------------------------

class DualSpaceType(object):
    LAGRANGE = S_(PETSCDUALSPACELAGRANGE)
    SIMPLE   = S_(PETSCDUALSPACESIMPLE)
    REFINED  = S_(PETSCDUALSPACEREFINED)
    BDM      = S_(PETSCDUALSPACEBDM)

# --------------------------------------------------------------------

cdef class DualSpace(Object):

    Type = DualSpaceType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.dualspace
        self.dualspace  = NULL

    def setUp(self):
        CHKERR( PetscDualSpaceSetUp(self.dualspace) )
    
    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDualSpace newdsp = NULL
        CHKERR( PetscDualSpaceCreate(ccomm, &newdsp) )
        PetscCLEAR(self.obj); self.dualspace = newdsp
        return self

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscDualSpaceView(self.dualspace, vwr) )

    def destroy(self):
        CHKERR( PetscDualSpaceDestroy(&self.dualspace) )
        return self

    def duplicate(self):
        cdef DualSpace spNew = DualSpace()
        CHKERR( PetscDualSpaceDuplicate(self.dualspace, &spNew.dualspace) )

    def getDM(self):
        cdef DM dm = DM()
        CHKERR( PetscDualSpaceGetDM(self.dualspace, &dm.dm) )
        return dm

    def setDM(self, DM dm):
        CHKERR( PetscDualSpaceSetDM(self.dualspace, dm.dm) )

    def getDimension(self):
        cdef PetscInt cdim
        CHKERR( PetscDualSpaceGetDimension(self.dualspace, &cdim))
        return toInt(cdim)   

    def getNumComponents(self):
        cdef PetscInt cncomps
        CHKERR( PetscDualSpaceGetNumComponents(self.dualspace, &cncomps))
        return toInt(cncomps)

    def setNumComponents(self, nc):
        cdef PetscInt cnc = asInt(nc)
        CHKERR( PetscDualSpaceSetNumComponents(self.dualspace, cnc) )    
    
    def getType(self):
        cdef PetscDualSpaceType cval = NULL
        CHKERR( PetscDualSpaceGetType(self.dualspace, &cval) )
        return bytes2str(cval)
    
    def setType(self, dualspace_type):
        cdef PetscDualSpaceType cval = NULL
        space_type = str2bytes(dualspace_type, &cval)
        CHKERR( PetscDualSpaceSetType(self.dualspace, cval) )
        return self

    def getOrder(self):
        cdef PetscInt corder
        CHKERR( PetscDualSpaceGetOrder(self.dualspace, &corder))
        return toInt(corder)
    
    def setOrder(self, order):
        cdef PetscInt corder = asInt(order)
        CHKERR( PetscDualSpaceSetOrder(self.dualspace, corder) )

    def getNumDof(self):
        cdef const PetscInt *cndof = NULL
        cdef PetscInt cdim = 0
        CHKERR( PetscDualSpaceGetDimension(self.dualspace, &cdim) )
        CHKERR( PetscDualSpaceGetNumDof(self.dualspace, &cndof) )
        return array_i(cdim + 1, cndof)

    def getFunctional(self, i):
        cdef PetscInt ci = asInt(i)
        cdef Quad functional = Quad()
        CHKERR( PetscDualSpaceGetFunctional( self.dualspace, ci, &functional.quad) )
        return functional

    def getInteriorDimension(self, intdim):
        cdef PetscInt cintdim = asInt(intdim)
        CHKERR( PetscDualSpaceGetInteriorDimension(self.dualspace, &cintdim) )
        return toInt(cintdim)

    def getLagrangeContinuity(self):
        cdef PetscBool ccontinuous = PETSC_FALSE
        CHKERR( PetscDualSpaceLagrangeGetContinuity(self.dualspace, &ccontinuous))
        return toBool(ccontinuous)
    
    def setLagrangeContinuity(self, continuous):
        cdef PetscBool ccontinuous = asBool(continuous)
        CHKERR( PetscDualSpaceLagrangeSetContinuity(self.dualspace, ccontinuous))
    
    def getLagrangeTensor(self):
        cdef PetscBool ctensor = PETSC_FALSE
        CHKERR( PetscDualSpaceLagrangeGetTensor(self.dualspace, &ctensor))
        return toBool(ctensor)
    
    def setLagrangeTensor(self, tensor):
        cdef PetscBool ctensor = asBool(tensor)
        CHKERR( PetscDualSpaceLagrangeSetTensor(self.dualspace, ctensor))

    def getLagrangeTrimmed(self):
        cdef PetscBool ctrimmed = PETSC_FALSE
        CHKERR( PetscDualSpaceLagrangeGetTrimmed(self.dualspace, &ctrimmed))
        return toBool(ctrimmed)
    
    def setLagrangeTrimmed(self, trimmed):
        cdef PetscBool ctrimmed = asBool(trimmed)
        CHKERR( PetscDualSpaceLagrangeSetTrimmed(self.dualspace, ctrimmed))
    
    def viewFromOptions(self, name, Object obj=None):
        cdef const char *cname = NULL
        _ = str2bytes(name, &cname)
        cdef PetscObject  cobj = NULL
        if obj is not None: cobj = obj.obj[0]
        CHKERR( PetscDualSpaceViewFromOptions(self.dualspace, cobj, cname) )

    def setSimpleDimension(self, dim):
        cdef PetscInt cdim = asInt(dim)
        CHKERR( PetscDualSpaceSimpleSetDimension(self.dualspace, cdim) )

    def setSimpleFunctional(self, func, Quad functional):
        cdef PetscInt cfunc = asInt(func)
        CHKERR( PetscDualSpaceSimpleSetFunctional(self.dualspace, cfunc, functional.quad) )

del SpaceType
del DualSpaceType