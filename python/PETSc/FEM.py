#!/usr/bin/env python
import user
import importer
import script

class QuadratureGenerator(script.Script):
  def __init__(self):
    script.Script.__init__(self)
    import os
    self.baseDir = os.getcwd()
    return

  def setupPaths(self):
    import sys, os

    petscDir = os.getenv('PETSC_DIR')
    sys.path.append(os.path.join(petscDir, 'externalpackages', 'FIAT-0.2.5a'))
    sys.path.append(os.path.join(petscDir, 'externalpackages', 'ffc-0.2.3'))
    sys.path.append(os.path.join(petscDir, 'externalpackages', 'Generator'))
    return

  def setup(self):
    script.Script.setup(self)
    self.setupPaths()
    import Cxx, CxxHelper
    self.Cxx = CxxHelper.Cxx()
    return

  def createElement(self, shape, k):
    import FIAT.Lagrange
    return FIAT.Lagrange.Lagrange(shape, k)

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

    self.logPrint('Generating quadrature structures for degree '+str(degree))
    ext = '_'+str(num)
    numPoints = Define()
    numPoints.identifier = 'NUM_QUADRATURE_POINTS'+ext
    numPoints.replacementText = str(len(quadrature.get_points()))
    return [numPoints,
            self.getArray(self.Cxx.getVar('points'+ext), quadrature.get_points(), 'Quadrature points\n   - (x1,y1,x2,y2,...)'),
            self.getArray(self.Cxx.getVar('weights'+ext), quadrature.get_weights(), 'Quadrature weights\n   - (v1,v2,...)')]

  def getBasisStructs(self, name, element, quadrature, num):
    '''Return C arrays with the basis functions and their derivatives evalauted at the quadrature points
       - FIAT uses a reference element of (-1,-1):(1,-1):(-1,1)'''
    from Cxx import Define
    import FIAT.shapes
    import Numeric

    points = quadrature.get_points()
    basis = element.function_space()
    dim = FIAT.shapes.dimension(basis.base.shape)
    ext = '_'+str(num)
    numFunctions = Define()
    numFunctions.identifier = 'NUM_BASIS_FUNCTIONS'+ext
    numFunctions.replacementText = str(len(basis))
    basisName = name+'Basis'+ext
    basisDerName = name+'BasisDerivatives'+ext
    return [numFunctions,
            self.getArray(self.Cxx.getVar(basisName), Numeric.transpose(basis.tabulate(points)), 'Nodal basis function evaluations\n    - basis function is fastest varying, then point'),
            self.getArray(self.Cxx.getVar(basisDerName), Numeric.transpose([basis.deriv_all(d).tabulate(points) for d in range(dim)]), 'Nodal basis function derivative evaluations,\n    - derivative direction fastest varying, then basis function, then point')]

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
    from Compiler import CodePurpose
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
    from Compiler import CompilerException

    self.logPrint('Generating element module')
    try:
      defns = []
      for n, (shape, k) in enumerate(elements):
        for quadrature in [self.createQuadrature(shape, 2*k+1)]:
          defns.extend(self.getQuadratureStructs(quadrature.degree, quadrature, n))
        for element, quadrature in [(self.createElement(shape, k), self.createQuadrature(shape, 2*k+1))]:
          #name = element.family+str(element.n)
          name = ''
          defns.extend(self.getBasisStructs(name, element, quadrature, n))
      #defns.extend(self.getQuadratureSetup())
      #defns.extend(self.getElementIntegrals())
    except CompilerException, e:
      print e
      raise RuntimeError('Quadrature source generation failed')
    return defns

  def outputElementSource(self, defns, filename = ''):
    from Compiler import CodePurpose
    import CxxVisitor

    # May need to move setupPETScLogging() here because PETSc clients are currently interfering with Numeric
    source = {'Cxx': [self.getQuadratureFile(filename, defns)]}
    outputs = {'Cxx': CxxVisitor.Output()}
    self.logPrint('Writing element source')
    for language,output in outputs.items():
      output.setRoot(CodePurpose.STUB, self.baseDir)
      output.setRoot(CodePurpose.IOR, self.baseDir)
      output.setRoot(CodePurpose.SKELETON, self.baseDir)
      try:
        map(lambda tree: tree.accept(output), source[language])
        for f in output.getFiles():
          self.logPrint('Created '+str(language)+' file '+str(f))
      except RuntimeError, e:
        print e
    return

  def run(self):
    self.setup()
    import FIAT.shapes
    order = 1
    self.logPrint('Making a P'+str(order)+' element')
    self.outputElementSource(self.getElementSource([(self.shape, self.order)]))
    return

if __name__ == '__main__':
  QuadratureGenerator().run()
