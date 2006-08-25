#!/usr/bin/env python
import user
import importer
import script

class P1(script.Script):
  def __init__(self):
    script.Script.__init__(self)
    self.setupPaths()
    import Cxx, CxxHelper
    import FIAT.shapes
    import os

    self.Cxx = CxxHelper.Cxx()
    self.baseDir = os.getcwd()


    # -------------------------------
    self.shape = FIAT.shapes.TRIANGLE
    self.order = 2
    # -------------------------------
    
    return

  def setupPaths(self):
    import sys

    sys.path.append('/home/ecoon/applications/petsc-dev/python:/home/ecoon/applications/python_modules/c-generator')
    sys.path.append('/home/ecoon/applications/FIAT-0.2.3')
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

  def getQuadratureStructs(self, degree, quadrature, mangle = 1):
    '''Return C arrays with the quadrature points and weights
       - FIAT uses a reference element of (-1,-1):(1,-1):(-1,1)'''
    from Cxx import Define

    self.logPrint('Generating quadrature structures for degree '+str(degree))
    numPoints = Define()
    numPoints.identifier = 'NUM_QUADRATURE_POINTS'
    numPoints.replacementText = str(len(quadrature.get_points()))
    if mangle:
      ext = str(degree)
    else:
      ext = ''
    return [numPoints,
            self.getArray(self.Cxx.getVar('points'+ext), quadrature.get_points(), 'Quadrature points\n   - (x1,y1,x2,y2,...)'),
            self.getArray(self.Cxx.getVar('weights'+ext), quadrature.get_weights(), 'Quadrature weights\n   - (v1,v2,...)')]

  def getBasisStructs(self, name, element, quadrature, mangle = 1):
    '''Return C arrays with the basis functions and their derivatives evalauted at the quadrature points
       - FIAT uses a reference element of (-1,-1):(1,-1):(-1,1)'''
    from Cxx import Define
    import FIAT.shapes
    import Numeric

    points = quadrature.get_points()
    basis = element.function_space()
    dim = FIAT.shapes.dimension(basis.base.shape)
    numFunctions = Define()
    numFunctions.identifier = 'NUM_BASIS_FUNCTIONS'
    numFunctions.replacementText = str(len(basis))
    if mangle:
      basisName = name+'Basis'+str(quadrature.degree)
      basisDerName = name+'BasisDerivatives'+str(quadrature.degree)
    else:
      basisName = name+'Basis'
      basisDerName = name+'BasisDerivatives'

    
    return [numFunctions,
            self.getArray(self.Cxx.getVar(basisName), Numeric.transpose(basis.tabulate(points)), 'Nodal basis function evaluations\n    - basis function is fastest varying, then point'),
            self.getArray(self.Cxx.getVar(basisDerName), Numeric.transpose([basis.deriv_all(d).tabulate(points) for d in range(dim)]), 'Nodal basis function derivative evaluations,\n    - derivative direction fastest varying, then basis function, then point')]

  def getSkeletonFile(self, basename, decls):
    from Compiler import CodePurpose
    from Cxx import Include
    from Cxx import Source
    import os

    # Needed to define NULL
    stdInclude = Include()
    stdInclude.identifier = '<stdlib.h>'
    source     = Source()
    source.filename = 'Integration.c'
    if basename:
      source.filename = basename+'_'+name
    source.children = [stdInclude]+decls
    source.purpose  = CodePurpose.SKELETON
    return source

  def getElementSource(self, shape, k):
    from Compiler import CompilerException

    self.logPrint('Generating element module')
    source = {'Cxx': []}
    try:
      defns = []
      for quadrature in [self.createQuadrature(shape, 2*k+1)]:
        defns.extend(self.getQuadratureStructs(quadrature.degree, quadrature, mangle = 0))
      for element, quadrature in [(self.createElement(shape, k), self.createQuadrature(shape, 2*k+1))]:
        #name = element.family+str(element.n)
        name = ''
        defns.extend(self.getBasisStructs(name, element, quadrature, mangle = 0))
      source['Cxx'].append(self.getSkeletonFile(name, defns))
    except CompilerException, e:
      print e
      raise RuntimeError('Quadrature source generation failed')
    return source

  def outputElementSource(self, shape, k):
    from Compiler import CodePurpose
    import CxxVisitor

    # May need to move setupPETScLogging() here because PETSc clients are currently interfering with Numeric
    outputs = {'Cxx':CxxVisitor.Output()}
    source  = self.getElementSource(shape, k)
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
    del source
    return
  
  def run(self):
    self.setup()
    self.logPrint('Making a P1 element')
    self.outputElementSource(self.shape, self.order)
    return

if __name__ == '__main__':
  P1().run()
