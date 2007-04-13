#!/usr/bin/env python
import user
import importer
import script

class QuadratureGenerator(script.Script):
  def __init__(self):
    import RDict
    script.Script.__init__(self, argDB = RDict.RDict())
    import os
    self.baseDir = os.getcwd()
    return

  def setupPaths(self):
    import sys, os

    petscDir = os.getenv('PETSC_DIR')
    sys.path.append(os.path.join(petscDir, 'externalpackages', 'fiat-dev'))
    sys.path.append(os.path.join(petscDir, 'externalpackages', 'ffc-0.2.3'))
    sys.path.append(os.path.join(petscDir, 'externalpackages', 'Generator'))
    return

  def setup(self):
    script.Script.setup(self)
    self.setupPaths()
    import Cxx, CxxHelper
    self.Cxx = CxxHelper.Cxx()
    if len(self.debugSections) == 0:
      self.debugSections = ['screen']
    return

  def createQuadrature(self, shape, degree):
    import FIAT.quadrature
    return FIAT.quadrature.make_quadrature_by_degree(shape, degree)

  def getArray(self, name, values, comment = None, typeName = 'double'):
    from Cxx import Array
    from Cxx import Initializer
    import Numeric

    values = Numeric.array(values)
    arrayInit = Initializer()
    arrayInit.children = map(self.Cxx.getDouble, Numeric.ravel(values))
    arrayInit.list = True
    arrayDecl = Array()
    arrayDecl.children = [name]
    arrayDecl.type = self.Cxx.typeMap[typeName]
    arrayDecl.size = self.Cxx.getInteger(Numeric.size(values))
    arrayDecl.static = True
    arrayDecl.initializer = arrayInit
    return self.Cxx.getDecl(arrayDecl, comment)

  def getQuadratureStructs(self, degree, quadrature, num):
    '''Return C arrays with the quadrature points and weights
       - FIAT uses a reference element of (-1,-1):(1,-1):(-1,1)'''
    from Cxx import Define

    self.logPrint('Generating quadrature structures for degree '+str(degree), debugSection = 'codegen')
    ext = '_'+str(num)
    numPoints = Define()
    numPoints.identifier = 'NUM_QUADRATURE_POINTS'+ext
    numPoints.replacementText = str(len(quadrature.get_points()))
    return [numPoints,
            self.getArray(self.Cxx.getVar('points'+ext), quadrature.get_points(), 'Quadrature points\n   - (x1,y1,x2,y2,...)'),
            self.getArray(self.Cxx.getVar('weights'+ext), quadrature.get_weights(), 'Quadrature weights\n   - (v1,v2,...)')]

  def getBasisFuncOrder(self, element):
    '''Map from FIAT order to Sieve order
       - In 2D, FIAT uses the numbering, and in 3D
         v2                                     v2
         |\                                     |\
         | \                                    |\\
       e1|  \e0                                 | |\
         |   \                                e1| | \e0
         v0--v1                                 | \  \
           e2                                   |  |e5\
                                                |  |   \
                                                |  |    \
                                                |  v4    \
                                                | /  \e4  \
                                                | |e3 ----\\
                                                |/         \\
                                                v0-----------v1
                                                    e2
    '''
    import FIAT.shapes
    basis = element.function_space()
    dim   = FIAT.shapes.dimension(basis.base.shape)
    ids   = element.Udual.entity_ids
    if dim == 1:
      perm = []
      for e in ids[1]:
        perm.extend(ids[1][e])
      for v in ids[0]:
        perm.extend(ids[0][v])
    elif dim == 2:
      perm = []
      for f in ids[2]:
        perm.extend(ids[2][f])
      for e in ids[1]:
        perm.extend(ids[1][(e+2)%3])
      for v in ids[0]:
        perm.extend(ids[0][v])
    elif dim == 3:
      print 'WARING: 3D FIAT-Sieve permuation not yet implemented'
      perm = []
      for c in ids[3]:
        perm.extend(ids[3][c])
      for f in ids[2]:
        perm.extend(ids[2][f])
      for e in ids[1]:
        perm.extend(ids[1][(e+2)%6])
      for v in ids[0]:
        perm.extend(ids[0][v])
    else:
      perm = None
    print element.Udual.pts
    print element.Udual.entity_ids
    print 'Perm:',perm
    return perm

  def getBasisStructs(self, name, element, quadrature, num):
    '''Return C arrays with the basis functions and their derivatives evalauted at the quadrature points
       - FIAT uses a reference element of (-1,-1):(1,-1):(-1,1)'''
    from Cxx import Define
    import FIAT.shapes
    import Numeric

    self.logPrint('Generating basis structures for element '+str(element.__class__), debugSection = 'codegen')
    points = quadrature.get_points()
    basis = element.function_space()
    dim = FIAT.shapes.dimension(basis.base.shape)
    ext = '_'+str(num)
    numFunctions = Define()
    numFunctions.identifier = 'NUM_BASIS_FUNCTIONS'+ext
    numFunctions.replacementText = str(len(basis))
    basisName = name+'Basis'+ext
    basisDerName = name+'BasisDerivatives'+ext
    perm = self.getBasisFuncOrder(element)
    basisTab = Numeric.transpose(basis.tabulate(points))
    basisDerTab = Numeric.transpose([basis.deriv_all(d).tabulate(points) for d in range(dim)])
    if not perm is None:
      basisTabOld    = Numeric.array(basisTab)
      basisDerTabOld = Numeric.array(basisDerTab)
      for q in range(len(points)):
        for i,pi in enumerate(perm):
          basisTab[q][i]    = basisTabOld[q][pi]
          basisDerTab[q][i] = basisDerTabOld[q][pi]
    return [numFunctions,
            self.getArray(self.Cxx.getVar(basisName), basisTab, 'Nodal basis function evaluations\n    - basis function is fastest varying, then point'),
            self.getArray(self.Cxx.getVar(basisDerName), basisDerTab, 'Nodal basis function derivative evaluations,\n    - derivative direction fastest varying, then basis function, then point')]

  def getQuadratureBlock(self, num):
    from Cxx import CompoundStatement
    cmpd = CompoundStatement()
    names = [('numQuadPoints', 'NUM_QUADRATURE_POINTS_'+str(num)),
             ('quadPoints',    'points_'+str(num)),
             ('quadWeights',   'weights_'+str(num)),
             ('numBasisFuncs', 'NUM_BASIS_FUNCTIONS_'+str(num)),
             ('basis',         'Basis_'+str(num)),
             ('basisDer',      'BasisDerivatives_'+str(num))]
    cmpd.children = [self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getIndirection(self.Cxx.getVar(a)), self.Cxx.getVar(b))) for a, b in names]
    return cmpd

  def getQuadratureSetup(self):
    from Cxx import Equality
    from Cxx import If
    funcName = 'setupUnstructuredQuadrature_gen'
    stmts = []
    stmts.append(self.Cxx.getExpStmt(self.Cxx.getVar('PetscFunctionBegin')))
    dimVar = self.Cxx.getVar('dim')
    dim3 = If()
    dim3.branch = self.Cxx.getEquality(dimVar, self.Cxx.getInteger(3))
    dim3.children = [self.getQuadratureBlock(2), self.Cxx.getPetscError('PETSC_ERR_SUP', 'Dimension not supported: %d', 'dim')]
    dim2 = If()
    dim2.branch = self.Cxx.getEquality(dimVar, self.Cxx.getInteger(2))
    dim2.children = [self.getQuadratureBlock(1), dim3]
    dim1 = If()
    dim1.branch = self.Cxx.getEquality(dimVar, self.Cxx.getInteger(1))
    dim1.children = [self.getQuadratureBlock(0), dim2]
    stmts.append(dim1)
    stmts.append(self.Cxx.getReturn(isPetsc = 1))
    func = self.Cxx.getFunction(funcName, self.Cxx.getType('PetscErrorCode'),
                                [self.Cxx.getParameter('dim', self.Cxx.getTypeMap()['const int']),
                                 self.Cxx.getParameter('numQuadPoints', self.Cxx.getTypeMap()['int pointer']),
                                 self.Cxx.getParameter('quadPoints', self.Cxx.getTypeMap()['double pointer pointer']),
                                 self.Cxx.getParameter('quadWeights', self.Cxx.getTypeMap()['double pointer pointer']),
                                 self.Cxx.getParameter('numBasisFuncs', self.Cxx.getTypeMap()['int pointer']),
                                 self.Cxx.getParameter('basis', self.Cxx.getTypeMap()['double pointer pointer']),
                                 self.Cxx.getParameter('basisDer', self.Cxx.getTypeMap()['double pointer pointer'])],
                                [], stmts)
    return self.Cxx.getFunctionHeader(funcName)+[func]

  def getIntegratorSetup(self, n, element):
    import FIAT.shapes
    from Cxx import Break, CompoundStatement, Function, Pointer, Switch
    dim  = FIAT.shapes.dimension(element.function_space().base.shape)
    ids  = element.Udual.entity_ids
    pts  = element.Udual.pts
    perm = self.getBasisFuncOrder(element)
    p    = 0
    funcName = 'IntegrateDualBasis_gen_'+str(n)
    idxVar  = self.Cxx.getVar('dualIndex')
    refVar  = self.Cxx.getVar('refCoords')
    decls = []
    decls.append(self.Cxx.getArray(refVar,   self.Cxx.getType('double'), 2))
    decls.append(self.Cxx.getArray('coords', self.Cxx.getType('double'), 2))
    stmts = []
    switch  = Switch()
    switch.branch = idxVar
    cmpd = CompoundStatement()
    if dim == 1:
##      if len(ids[1][0]) == 0:
##        cmpd.children.append(self.Cxx.getReturn(0.0, caseLabel = p))
##      if len(ids[0][0]) == 0:
##        retStmt = self.Cxx.getReturn(0.0, caseLabel = p)
##        retStmt.caseLabel = self.Cxx.getValue('1')
##        cmpd.children.append(retStmt)
##        retStmt = self.Cxx.getReturn(0.0, caseLabel = p)
##        retStmt.caseLabel = self.Cxx.getValue('2')
##        cmpd.children.append(retStmt)
      for i in range(len(ids[1][0]) + len(ids[0][0])*2):
        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 0), pts[perm[p]][0]), caseLabel = p)
        cmpd.children.extend([cStmt, Break()])
        p += 1
    elif dim == 2:
##      if len(ids[2][0]) == 0:
##        cmpd.children.append(self.Cxx.getReturn(0.0, caseLabel = p))
##      elif len(ids[2][0]) == 1:
##        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 0), pts[perm[p]][0]), caseLabel = p)
##        cmpd.children.extend([cStmt, self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 1), pts[perm[p]][1])), Break()])
##        p += 1
##      if len(ids[1][0]) == 0:
##        cmpd.children.append(self.Cxx.getReturn(0.0, caseLabel = '1'))
##        cmpd.children.append(self.Cxx.getReturn(0.0, caseLabel = '2'))
##        cmpd.children.append(self.Cxx.getReturn(0.0, caseLabel = '3'))
##      elif len(ids[1][0]) == 1:
##        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 0), pts[perm[p]][0]))
##        cStmt.caseLabel = self.Cxx.getValue('1')
##        cmpd.children.extend([cStmt, self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 1), pts[perm[p]][1])), Break()])
##        p += 1
##        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 0), pts[perm[p]][0]))
##        cStmt.caseLabel = self.Cxx.getValue('2')
##        cmpd.children.extend([cStmt, self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 1), pts[perm[p]][1])), Break()])
##        p += 1
##        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 0), pts[perm[p]][0]))
##        cStmt.caseLabel = self.Cxx.getValue('3')
##        cmpd.children.extend([cStmt, self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 1), pts[perm[p]][1])), Break()])
##        p += 1
##      if len(ids[0][0]) == 0:
##        cmpd.children.append(self.Cxx.getReturn(0.0, caseLabel = '4'))
##        cmpd.children.append(self.Cxx.getReturn(0.0, caseLabel = '5'))
##        cmpd.children.append(self.Cxx.getReturn(0.0, caseLabel = '6'))
##      elif len(ids[0][0]) == 1:
##        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 0), pts[perm[p]][0]))
##        cStmt.caseLabel = self.Cxx.getValue('4')
##        cmpd.children.extend([cStmt, self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 1), pts[perm[p]][1])), Break()])
##        p += 1
##        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 0), pts[perm[p]][0]))
##        cStmt.caseLabel = self.Cxx.getValue('5')
##        cmpd.children.extend([cStmt, self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 1), pts[perm[p]][1])), Break()])
##        p += 1
##        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 0), pts[perm[p]][0]))
##        cStmt.caseLabel = self.Cxx.getValue('6')
##        cmpd.children.extend([cStmt, self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 1), pts[perm[p]][1])), Break()])
##        p += 1
      for i in range(len(ids[2][0]) + len(ids[1][0])*3 + len(ids[0][0])*3):
        cStmt = self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 0), pts[perm[p]][0]), caseLabel = p)
        cmpd.children.extend([cStmt, self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(refVar, 1), pts[perm[p]][1])), Break()])
        p += 1
    elif dim == 3:
      pass
    cStmt = self.Cxx.getExpStmt(self.Cxx.getFunctionCall('printf', [self.Cxx.getString('dualIndex: %d\\n'), 'dualIndex']))
    cStmt.caseLabel = self.Cxx.getValue('default')
    cmpd.children.extend([cStmt, self.Cxx.getThrow(self.Cxx.getFunctionCall('ALE::Exception', [self.Cxx.getString('Bad dual index')]))])
    switch.children = [cmpd]
    stmts.append(switch)
    basisLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('e', 'int'), 0, dim)
    basisLoop.children[0].children.append(self.Cxx.getExpStmt(self.Cxx.getAdditionAssignment(self.Cxx.getArrayRef('coords', 'd'), self.Cxx.getMultiplication(self.Cxx.getArrayRef('J', self.Cxx.getAddition(self.Cxx.getMultiplication('d', dim), 'e')), self.Cxx.getGroup(self.Cxx.getAddition(self.Cxx.getArrayRef(refVar, 'e'), 1.0))))))
    testLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('d', 'int'), 0, dim)
    testLoop.children[0].children.extend([self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef('coords', 'd'), self.Cxx.getArrayRef('v0', 'd'))), basisLoop])
    stmts.append(testLoop)
    stmts.append(self.Cxx.getReturn(self.Cxx.getFunctionCall(self.Cxx.getGroup(self.Cxx.getIndirection('func')), ['coords'])))
    bcFunc = Function()
    bcFunc.children = [self.Cxx.getDeclGroup(self.Cxx.getDeclarator('func', Pointer()))]
    bcFunc.type = self.Cxx.getType('double')
    bcFunc.parameters = [self.Cxx.getParameter('coords', self.Cxx.getType('double', 1, isConst = 1))]
    func = self.Cxx.getFunction(funcName, self.Cxx.getType('double'),
                                [self.Cxx.getParameter('v0', self.Cxx.getType('double', 1, isConst = 1)),
                                 self.Cxx.getParameter('J',  self.Cxx.getType('double', 1, isConst = 1)),
                                 self.Cxx.getParameter(idxVar, self.Cxx.getType('int', isConst = 1)),
                                 self.Cxx.getParameter(None, bcFunc)],
                                decls, stmts)
    return self.Cxx.getFunctionHeader(funcName)+[func]

  def getSectionSetup(self, n, element):
    from Cxx import CompoundStatement
    funcName = 'CreateProblem_gen_'+str(n)
    meshVar  = self.Cxx.getVar('mesh')
    secVar   = self.Cxx.getVar('section')
    optVar   = self.Cxx.getVar('options')
    decls = []
    decls.append(self.Cxx.getDeclaration(meshVar, self.Cxx.getType('Mesh'), self.Cxx.castToType('dm', self.Cxx.getType('Mesh'))))
    decls.append(self.Cxx.getDeclaration(secVar, self.Cxx.getType('SectionReal')))
    decls.append(self.Cxx.getDeclaration('m', self.Cxx.getType('ALE::Obj<ALE::Mesh>'), isForward=1))
    decls.append(self.Cxx.getDeclaration('s', self.Cxx.getType('ALE::Obj<ALE::Mesh::real_section_type>'), isForward=1))
    decls.append(self.Cxx.getDeclaration('ierr', self.Cxx.getType('PetscErrorCode')))
    stmts = []
    stmts.append(self.Cxx.getExpStmt(self.Cxx.getVar('PetscFunctionBegin')))
    stmts.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('MeshGetMesh', [meshVar, 'm'])))
    stmts.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('SectionRealCreate',
                                                                 [self.Cxx.getFunctionCall(self.Cxx.getStructRef('m', 'comm')),
                                                                  self.Cxx.getAddress(secVar)])))
    stmts.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('SectionRealSetBundle', [secVar, 'm'])))
    stmts.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('SectionRealGetSection', [secVar, 's'])))
    stmts.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscObjectSetName', [self.Cxx.castToType(secVar, self.Cxx.getType('PetscObject')),
                                                                                        self.Cxx.getString('default')])))
    stmts.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('MeshSetSectionReal', [meshVar, secVar])))
    cmpd = CompoundStatement()
    cmpd.declarations = [
      self.Cxx.getDeclaration('d', self.Cxx.getType('ALE::Obj<ALE::Discretization>&', isConst=1),
                              self.Cxx.getFunctionCall(self.Cxx.getStructRef('m', 'getDiscretization'))),
      self.Cxx.getDeclaration('b', self.Cxx.getType('ALE::Obj<ALE::BoundaryCondition>&', isConst=1),
                              self.Cxx.getFunctionCall(self.Cxx.getStructRef('m', 'getBoundaryCondition')))]
    for d, ids in element.Udual.entity_ids.items():
      cmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('d', 'setNumDof'), [d, len(ids[0])])))
    trueBranch = CompoundStatement()
    trueBranch.children = [self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('b', 'setLabelName'), [self.Cxx.getString('marker')])),
                           self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('b', 'setFunction'), [self.Cxx.getStructRef(optVar, 'exactFunc')])),
                           self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('b', 'setDualIntegrator'), ['IntegrateDualBasis_gen_'+str(n)]))]
    cmpd.children.append(self.Cxx.getIf(self.Cxx.getEquality(self.Cxx.getStructRef(optVar, 'bcType'), 'DIRICHLET'), [trueBranch]))
    cmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('s', 'setDebug'), [self.Cxx.getStructRef(optVar, 'debug')])))
    cmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('m', 'setupField'), ['s'])))
    cmpd.children.append(self.Cxx.getIf(self.Cxx.getFunctionCall(self.Cxx.getStructRef('m', 'debug')),
                                        [self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef('s', 'view'),
                                                                                      [self.Cxx.getString('Default field')]))]))
    stmts.append(cmpd)
    stmts.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('SectionRealDestroy', [secVar])))
    stmts.append(self.Cxx.getReturn(isPetsc = 1))
    func = self.Cxx.getFunction(funcName, self.Cxx.getType('PetscErrorCode'),
                                [self.Cxx.getParameter('dm', self.Cxx.getType('DM')),
                                 self.Cxx.getParameter(optVar, self.Cxx.getType('Options', 1))],
                                decls, stmts)
    return self.Cxx.getFunctionHeader(funcName)+[func]

  def getRealCoordinates(self, dimVar, v0Var, JVar, coordsVar):
    '''Calculates the real coordinates of each quadrature point from reference coordinates'''
    dim2Loop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('e', 'int'), 0, dimVar)
    dim2Loop.children[0].children.append(self.Cxx.getExpStmt(self.Cxx.getAdditionAssignment(self.Cxx.getArrayRef(coordsVar, 'd'), self.Cxx.getMultiplication(self.Cxx.getArrayRef(JVar, self.Cxx.getAddition(self.Cxx.getMultiplication('d', dimVar), 'e')), self.Cxx.getGroup(self.Cxx.getAddition(self.Cxx.getArrayRef('quadPoints', self.Cxx.getAddition(self.Cxx.getMultiplication('q', dimVar), 'e')), self.Cxx.getDouble(1.0)))))))
    dimLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('d', 'int'), 0, dimVar)
    dimLoop.children[0].children.append(self.Cxx.getExpStmt(self.Cxx.getAssignment(self.Cxx.getArrayRef(coordsVar, 'd'), self.Cxx.getArrayRef(v0Var, 'd'))))
    dimLoop.children[0].children.append(dim2Loop)
    return dimLoop

  def getLinearAccumulation(self, inputSection, outputSection, elemIter, numBasisFuncs, elemVec, elemMat):
    '''Accumulates linear terms in the residual'''
    from Cxx import CompoundStatement
    scope = CompoundStatement()
    scope.declarations.append(self.Cxx.getDeclaration('x', self.Cxx.getType('PetscScalar', 1)))
    scope.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('SectionRealRestrict', [inputSection, self.Cxx.getIndirection(elemIter), self.Cxx.getAddress('x')])))
    basisFuncLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('g', 'int'), 0, numBasisFuncs)
    basisFuncLoop.children[0].children.append(self.Cxx.getExpStmt(self.Cxx.getAdditionAssignment(self.Cxx.getArrayRef(elemVec, 'f'), self.Cxx.getMultiplication(self.Cxx.getArrayRef(elemMat, self.Cxx.getAddition(self.Cxx.getMultiplication('f', 'numBasisFuncs'), 'g')), self.Cxx.getArrayRef('x', 'g')))))
    testFuncLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('f', 'int'), 0, numBasisFuncs)
    testFuncLoop.children[0].children.append(basisFuncLoop)
    scope.children.append(testFuncLoop)
    scope.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('SectionRealUpdateAdd', [outputSection, self.Cxx.getIndirection(elemIter), elemVec])))
    return scope

  def getElementIntegrals(self):
    '''Output the C++ residual calculation method'''
    from Cxx import CompoundStatement
    from Cxx import DeclaratorGroup
    from Cxx import Declarator
    from Cxx import Function
    from Cxx import FunctionCall
    from Cxx import Pointer
    from Cxx import Type
    # Parameters
    ctxVar = self.Cxx.getVar('ctx')
    # C Declarations
    decls = []
    optionsType = self.Cxx.getType('Options', 1)
    optionsVar = self.Cxx.getVar('options')
    funcVar = self.Cxx.getVar('func')
    meshVar = self.Cxx.getVar('mesh')
    mVar = self.Cxx.getVar('m')
    decls.append(self.Cxx.getDeclaration(optionsVar, optionsType, self.Cxx.castToType(ctxVar, optionsType)))
    paramDecl = Declarator()
    paramDecl.type = self.Cxx.getType('double', isConst = 1, numPointers = 1)
    funcHead = DeclaratorGroup()
    #   This is almost certainly wrong
    funcHead.setChildren([self.Cxx.getIndirection(funcVar)])
    funcDecl = Function()
    funcDecl.setChildren([funcHead])
    funcDecl.type = self.Cxx.getType('PetscScalar')
    funcDecl.parameters = [paramDecl]
    funcDecl.initializer = self.Cxx.getStructRef(optionsVar, 'rhsFunc')
    decls.append(self.Cxx.getDecl(funcDecl))
    decls.append(self.Cxx.getDeclaration(mVar, self.Cxx.getType('ALE::Obj<ALE::Mesh>'), self.Cxx.getNullVar()))
    decls.extend([self.Cxx.getDeclaration(self.Cxx.getVar('numQuadPoints'), self.Cxx.getTypeMap()['int']),
                  self.Cxx.getDeclaration(self.Cxx.getVar('numBasisFuncs'), self.Cxx.getTypeMap()['int'])])
    decls.extend([self.Cxx.getDeclaration(self.Cxx.getVar('quadPoints'),  self.Cxx.getType('double pointer')),
                  self.Cxx.getDeclaration(self.Cxx.getVar('quadWeights'), self.Cxx.getType('double pointer')),
                  self.Cxx.getDeclaration(self.Cxx.getVar('basis'),       self.Cxx.getType('double pointer')),
                  self.Cxx.getDeclaration(self.Cxx.getVar('basisDer'),    self.Cxx.getType('double pointer'))])
    decls.append(self.Cxx.getDeclaration(self.Cxx.getVar('ierr'),  self.Cxx.getType('PetscErrorCode')))
    # C++ Declarations and Setup
    stmts = []
    stmts.append(self.Cxx.getExpStmt(self.Cxx.getVar('PetscFunctionBegin')))
    stmts.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('MeshGetMesh', [meshVar, mVar])))
    stmts.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('setupUnstructuredQuadrature_gen', [self.Cxx.getStructRef(optionsVar, 'dim'), self.Cxx.getAddress('numQuadPoints'), self.Cxx.getAddress('quadPoints'), self.Cxx.getAddress('quadWeights'), self.Cxx.getAddress('numBasisFuncs'), self.Cxx.getAddress('basis'), self.Cxx.getAddress('basisDer')])))
    patchVar = self.Cxx.getVar('patch')
    coordinatesVar = self.Cxx.getVar('coordinates')
    topologyVar = self.Cxx.getVar('topology')
    cellsVar = self.Cxx.getVar('cells')
    cornersVar = self.Cxx.getVar('corners')
    dimVar = self.Cxx.getVar('dim')
    t_derVar = self.Cxx.getVar('t_der')
    b_derVar = self.Cxx.getVar('b_der')
    coordsVar = self.Cxx.getVar('coords')
    v0Var = self.Cxx.getVar('v0')
    JVar = self.Cxx.getVar('J')
    invJVar = self.Cxx.getVar('invJ')
    detJVar = self.Cxx.getVar('detJ')
    elemVecVar = self.Cxx.getVar('elemVec')
    elemMatVar = self.Cxx.getVar('elemMat')
    decls.extend([self.Cxx.getDeclaration(t_derVar,   self.Cxx.getTypeMap()['double pointer']),
                  self.Cxx.getDeclaration(b_derVar,   self.Cxx.getTypeMap()['double pointer']),
                  self.Cxx.getDeclaration(coordsVar,  self.Cxx.getTypeMap()['double pointer']),
                  self.Cxx.getDeclaration(v0Var,      self.Cxx.getTypeMap()['double pointer']),
                  self.Cxx.getDeclaration(JVar,       self.Cxx.getTypeMap()['double pointer']),
                  self.Cxx.getDeclaration(invJVar,    self.Cxx.getTypeMap()['double pointer']),
                  self.Cxx.getDeclaration(detJVar,    self.Cxx.getTypeMap()['double']),
                  self.Cxx.getDeclaration(elemVecVar, self.Cxx.getType('PetscScalar', 1)),
                  self.Cxx.getDeclaration(elemMatVar, self.Cxx.getType('PetscScalar', 1))])
    cxxCmpd = CompoundStatement()
    cxxCmpd.declarations = [self.Cxx.getDeclaration(patchVar, self.Cxx.getType('ALE::Mesh::real_section_type::patch_type', 0, 1), self.Cxx.getInteger(0)),
                            self.Cxx.getDeclaration(coordinatesVar, self.Cxx.getType('ALE::Obj<ALE::Mesh::real_section_type>&', isConst = 1), self.Cxx.getFunctionCall(self.Cxx.getStructRef(mVar, 'getRealSection'), [self.Cxx.getString('coordinates')])),
                            self.Cxx.getDeclaration(topologyVar, self.Cxx.getType('ALE::Obj<ALE::Mesh::topology_type>&', isConst = 1), self.Cxx.getFunctionCall(self.Cxx.getStructRef(mVar, 'getTopology'))),
                            self.Cxx.getDeclaration(cellsVar, self.Cxx.getType('ALE::Obj<ALE::Mesh::topology_type::label_sequence>&', 0, 1), self.Cxx.getFunctionCall(self.Cxx.getStructRef(topologyVar, 'heightStratum'), [patchVar, 0])),
                            self.Cxx.getDeclaration(cornersVar, self.Cxx.getTypeMap()['const int'], self.Cxx.getFunctionCall(self.Cxx.getStructRef(self.Cxx.getFunctionCall(self.Cxx.getStructRef(self.Cxx.getFunctionCall(self.Cxx.getStructRef(topologyVar, 'getPatch'), [patchVar]), 'nCone'), [self.Cxx.getIndirection(self.Cxx.getFunctionCall(self.Cxx.getStructRef(cellsVar, 'begin'))), self.Cxx.getFunctionCall(self.Cxx.getStructRef(topologyVar, 'depth'))]), 'size'))),
                            self.Cxx.getDeclaration(dimVar, self.Cxx.getTypeMap()['const int'], self.Cxx.getFunctionCall(self.Cxx.getStructRef(mVar, 'getDimension')))]
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('SectionRealZero', ['section'])))
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscMalloc', [self.Cxx.getMultiplication(cornersVar, self.Cxx.getSizeof('PetscScalar')), self.Cxx.getAddress(elemVecVar)])))
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscMalloc', [self.Cxx.getMultiplication(cornersVar, self.Cxx.getMultiplication(cornersVar, self.Cxx.getSizeof('PetscScalar'))), self.Cxx.getAddress(elemMatVar)])))
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscMalloc6', [dimVar,'double',self.Cxx.getAddress(t_derVar),dimVar,'double',self.Cxx.getAddress(b_derVar),dimVar,'double',self.Cxx.getAddress(coordsVar),dimVar,'double',self.Cxx.getAddress(v0Var),self.Cxx.getMultiplication(dimVar,dimVar),'double',self.Cxx.getAddress(JVar),self.Cxx.getMultiplication(dimVar,dimVar),'double',self.Cxx.getAddress(invJVar)])))
    # Loop over elements
    lowerBound = FunctionCall()
    lowerBound.setChildren([self.Cxx.getStructRef(cellsVar, 'begin')])
    upperBound = FunctionCall()
    upperBound.setChildren([self.Cxx.getStructRef(cellsVar, 'end')])
    lType = Type()
    lType.identifier = 'ALE::Mesh::topology_type::label_sequence::iterator'
    decl = Declarator()
    decl.identifier = 'c_iter'
    decl.type = lType
    loop = self.Cxx.getSimpleLoop(decl, lowerBound, upperBound, allowInequality = 1, isPrefix = 1)
    loop.comments = [self.Cxx.getComment('Loop over elements')]
    loopCmpd = loop.children[0]
    # Loop body
    c_iterVar = self.Cxx.getVar('c_iter')
    loopCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscMemzero', [elemVecVar, self.Cxx.getMultiplication(cornersVar, self.Cxx.getSizeof('PetscScalar'))])))
    loopCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscMemzero', [elemMatVar, self.Cxx.getMultiplication(cornersVar, self.Cxx.getMultiplication(cornersVar, self.Cxx.getSizeof('PetscScalar')))])))
    loopCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getFunctionCall(self.Cxx.getStructRef(mVar, 'computeElementGeometry'), [coordinatesVar, self.Cxx.getIndirection(c_iterVar), v0Var, JVar, invJVar, detJVar])))
    #   Quadrature loop
    quadLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('q', 'int'), 0, 'numQuadPoints')
    quadLoop.comments = [self.Cxx.getComment('Loop over quadrature points')]
    quadCmpd = quadLoop.children[0]
    loopCmpd.children.append(quadLoop)
    #   Get real coordinates
    quadCmpd.children.append(self.getRealCoordinates(dimVar, v0Var, JVar, coordsVar))
    #   Accumulate constant terms
    funcValVar = self.Cxx.getVar('funcVal')
    quadCmpd.declarations.append(self.Cxx.getDeclaration(funcValVar, self.Cxx.getType('PetscScalar')))
    quadCmpd.children.append(self.Cxx.getExpStmt(self.Cxx.getAssignment(funcValVar, self.Cxx.getFunctionCall(self.Cxx.getGroup(self.Cxx.getIndirection(funcVar)), [coordsVar]))))
    testFuncLoop = self.Cxx.getSimpleLoop(self.Cxx.getDeclarator('f', 'int'), 0, 'numBasisFuncs')
    testFuncLoop.children[0].children.append(self.Cxx.getExpStmt(self.Cxx.getSubtractionAssignment(self.Cxx.getArrayRef(elemVecVar, 'f'), self.Cxx.getMultiplication(self.Cxx.getMultiplication(self.Cxx.getMultiplication(self.Cxx.getArrayRef('basis', self.Cxx.getAddition(self.Cxx.getMultiplication('q', 'numBasisFuncs'), 'f')), funcValVar), self.Cxx.getArrayRef('quadWeights', 'q')), detJVar))))
    quadCmpd.children.append(testFuncLoop)
    #   Accumulate linear terms
    loopCmpd.children.append(self.getLinearAccumulation('X', 'section', c_iterVar, 'numBasisFuncs', elemVecVar, elemMatVar))
    cxxCmpd.children.append(loop)
    # Cleanup
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscFree', [elemVecVar])))
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscFree', [elemMatVar])))
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('PetscFree6', [t_derVar,b_derVar,coordsVar,v0Var,JVar,invJVar])))
    cxxCmpd.children.extend(self.Cxx.getPetscCheck(self.Cxx.getFunctionCall('SectionRealComplete', ['section'])))
    # Pack up scopes
    stmts.append(cxxCmpd)
    stmts.append(self.Cxx.getReturn(isPetsc = 1))
    funcName = 'Rhs_Unstructured_gen'
    func = self.Cxx.getFunction(funcName, self.Cxx.getType('PetscErrorCode'),
                                [self.Cxx.getParameter('mesh', self.Cxx.getType('Mesh')),
                                 self.Cxx.getParameter('X', self.Cxx.getType('SectionReal')),
                                 self.Cxx.getParameter('section', self.Cxx.getType('SectionReal')),
                                 self.Cxx.getParameter('ctx', self.Cxx.getTypeMap()['void pointer'])],
                                decls, stmts)
    return self.Cxx.getFunctionHeader(funcName)+[func]

  def getQuadratureFile(self, filename, decls):
    from GenericCompiler import CodePurpose
    from Cxx import Include
    from Cxx import Header
    import os

    # Needed to define NULL
    stdInclude = Include()
    stdInclude.identifier = '<stdlib.h>'
    header            = Header()
    if filename:
      header.filename = filename
    else:
      header.filename = 'Integration.c'
    header.children   = [stdInclude]+decls
    header.purpose    = CodePurpose.SKELETON
    return header

  def getElementSource(self, elements):
    from GenericCompiler import CompilerException

    self.logPrint('Generating element module', debugSection = 'codegen')
    try:
      defns = []
      for n, element in enumerate(elements):
        #name      = element.family+str(element.n)
        name       = ''
        shape      = element.shape
        order      = element.order
        quadrature = self.createQuadrature(shape, 2*order+1)
        defns.extend(self.getQuadratureStructs(quadrature.degree, quadrature, n))
        defns.extend(self.getBasisStructs(name, element, quadrature, n))
        defns.extend(self.getIntegratorSetup(n, element))
        defns.extend(self.getSectionSetup(n, element))
      #defns.extend(self.getQuadratureSetup())
      #defns.extend(self.getElementIntegrals())
    except CompilerException, e:
      print e
      raise RuntimeError('Quadrature source generation failed')
    return defns

  def outputElementSource(self, defns, filename = ''):
    from GenericCompiler import CodePurpose
    import CxxVisitor

    # May need to move setupPETScLogging() here because PETSc clients are currently interfering with Numeric
    source = {'Cxx': [self.getQuadratureFile(filename, defns)]}
    outputs = {'Cxx': CxxVisitor.Output()}
    self.logPrint('Writing element source', debugSection = 'codegen')
    for language,output in outputs.items():
      output.setRoot(CodePurpose.STUB, self.baseDir)
      output.setRoot(CodePurpose.IOR, self.baseDir)
      output.setRoot(CodePurpose.SKELETON, self.baseDir)
      try:
        map(lambda tree: tree.accept(output), source[language])
        for f in output.getFiles():
          self.logPrint('Created '+str(language)+' file '+str(f), debugSection = 'codegen')
      except RuntimeError, e:
        print e
    return

  def run(self, elements, filename = ''):
    if elements is None:
      import FIAT.shapes
      import FIAT.Lagrange
      elements =[FIAT.Lagrange.Lagrange(FIAT.shapes.TRIANGLE, 1)]
      self.logPrint('Making a P'+str(order)+' element on a triangle')
    self.outputElementSource(self.getElementSource(elements), filename)
    return

if __name__ == '__main__':
  QuadratureGenerator().run()
