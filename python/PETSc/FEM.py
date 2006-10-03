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
    sys.path.append(os.path.join(petscDir, 'externalpackages', 'fiat-0.2.3'))
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
