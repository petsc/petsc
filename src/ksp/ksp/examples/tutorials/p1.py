#!/usr/bin/env python
import user
import importer
import script

class P1(script.Script):
  def __init__(self):
    script.Script.__init__(self)
    self.setupPaths()
    import ASE.Compiler.Python.Cxx
    import FIAT.shapes
    import os

    self.Cxx = ASE.Compiler.Python.Cxx.Cxx()
    self.baseDir = os.getcwd()
    self.shape = FIAT.shapes.TRIANGLE
    return

  def setupPaths(self):
    import sys

    sys.path.append('/PETSc3/fenics/fiat-cvs')
    sys.path.append('/PETSc3/ase/Runtime/client-python')
    sys.path.append('/PETSc3/ase/Runtime/server-python-ase')
    sys.path.append('/PETSc3/ase/Compiler/client-python')
    sys.path.append('/PETSc3/ase/Compiler/server-python-bootstrap')
    sys.path.append('/PETSc3/ase/Compiler/server-python-compiler')
    sys.path.append('/PETSc3/ase/Compiler/server-python-cxxI')
    sys.path.append('/PETSc3/ase/Compiler/server-python-cxxII')
    sys.path.append('/PETSc3/ase/Compiler/server-python-cxxIII')
    sys.path.append('/PETSc3/ase/Compiler/server-python-cxxIV')
    sys.path.append('/PETSc3/ase/Compiler/server-python-cxxV')
    sys.path.append('/PETSc3/ase/Compiler/server-python-cxxVisitor')
    sys.path.append('/PETSc3/ase/Compiler/server-python-pythonI')
    sys.path.append('/PETSc3/ase/Compiler/server-python-pythonII')
    sys.path.append('/PETSc3/ase/Compiler/server-python-pythonIII')
    sys.path.append('/PETSc3/ase/Compiler/server-python-pythonVisitor')
    sys.path.append('/PETSc3/ase/Compiler/server-python-sidl')
    sys.path.append('/PETSc3/ase/Compiler/server-python-sidlVisitor')

    import ASE.Loader
    ASE.Loader.Loader.setSearchPath(['/PETSc3/ase/Runtime/lib', '/PETSc3/ase/Compiler/lib'])
    return

  def createElement(self, shape, k):
    import FIAT.Lagrange
    return FIAT.Lagrange.Lagrange(shape, k)

  def createQuadrature(self, shape, degree):
    import FIAT.quadrature
    return FIAT.quadrature.make_quadrature_by_degree(shape, degree)

  def getArray(self, name, values, comment = None, typeName = 'double'):
    import ASE.Compiler.Cxx.Array
    import ASE.Compiler.Cxx.Initializer
    import Numeric

    values = Numeric.array(values)
    arrayInit = ASE.Compiler.Cxx.Initializer.Initializer()
    arrayInit.addChildren(map(self.Cxx.getDouble, Numeric.ravel(values)))
    arrayInit.setList(1)
    arrayDecl = ASE.Compiler.Cxx.Array.Array()
    arrayDecl.setChildren([name])
    arrayDecl.setType(self.Cxx.typeMap[typeName])
    arrayDecl.setSize(self.Cxx.getInteger(Numeric.size(values)))
    arrayDecl.setStatic(1)
    arrayDecl.setInitializer(arrayInit)
    return self.Cxx.getDecl(arrayDecl, comment)

  def getQuadratureStructs(self, degree, quadrature, mangle = 1):
    '''Return C arrays with the quadrature points and weights
       - FIAT uses a reference element of (-1,-1):(1,-1):(-1,1)'''
    import ASE.Compiler.Cxx.Define

    self.logPrint('Generating quadrature structures for degree '+str(degree))
    numPoints = ASE.Compiler.Cxx.Define.Define()
    numPoints.setIdentifier('NUM_QUADRATURE_POINTS')
    numPoints.setReplacementText(str(len(quadrature.get_points())))
    if mangle:
      ext = str(degree)
    else:
      ext = ''
    return [numPoints,
            self.getArray(self.Cxx.getVar('points'+ext), quadrature.get_points(), 'Quadrature points'),
            self.getArray(self.Cxx.getVar('weights'+ext), quadrature.get_weights(), 'Quadrature weights')]

  def getBasisStructs(self, name, element, quadrature, mangle = 1):
    '''Return C arrays with the basis functions and their derivatives evalauted at the quadrature points
       - FIAT uses a reference element of (-1,-1):(1,-1):(-1,1)'''
    import ASE.Compiler.Cxx.Define
    import FIAT.shapes
    import Numeric

    points = quadrature.get_points()
    basis = element.function_space()
    dim = FIAT.shapes.dimension(basis.base.shape)
    numFunctions = ASE.Compiler.Cxx.Define.Define()
    numFunctions.setIdentifier('NUM_BASIS_FNUCTIONS')
    numFunctions.setReplacementText(str(len(basis)))
    if mangle:
      basisName = name+'Basis'+str(quadrature.degree)
      basisDerName = name+'BasisDerivatives'+str(quadrature.degree)
    else:
      basisName = name+'Basis'
      basisDerName = name+'BasisDerivatives'
    return [numFunctions,
            self.getArray(self.Cxx.getVar(basisName), Numeric.transpose(basis.tabulate(points)), 'Nodal basis function evaluations'),
            self.getArray(self.Cxx.getVar(basisDerName), Numeric.transpose([basis.deriv_all(d).tabulate(points) for d in range(dim)]), 'Nodal basis function derivative evaluations')]

  def getSkeletonFile(self, basename, decls):
    import ASE.Compiler.CodePurpose
    import ASE.Compiler.Cxx.Include
    import ASE.Compiler.Cxx.Source
    import os

    # Needed to define NULL
    stdInclude = ASE.Compiler.Cxx.Include.Include()
    stdInclude.setIdentifier('<stdlib.h>')
    source     = ASE.Compiler.Cxx.Source.Source()
    name       = 'Integration.c'
    if basename:
      name     = basename+'_'+name
    source.setFilename(name)
    source.addChildren([stdInclude])
    source.addChildren(decls)
    source.setPurpose(ASE.Compiler.CodePurpose.SKELETON)
    return source

  def getElementSource(self, shape, k):
    import ASE.Compiler.CompilerException

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
    except ASE.Compiler.CompilerException.Exception, e:
      print e.getMessage()
      raise RuntimeError('Quadrature source generation failed')
    return source

  def outputElementSource(self, shape, k):
    import ASE.BaseException
    import ASE.Compiler.CodePurpose
    import ASE.Compiler.Cxx.Output

    # May need to move setupPETScLogging() here because PETSc clients are currently interfering with Numeric
    outputs = {'Cxx':ASE.Compiler.Cxx.Output.Output()}
    source  = self.getElementSource(shape, k)
    self.logPrint('Writing element source')
    for language,output in outputs.items():
      import gc
      gc.collect()
      output.setRoot(ASE.Compiler.CodePurpose.STUB, self.baseDir)
      output.setRoot(ASE.Compiler.CodePurpose.IOR, self.baseDir)
      output.setRoot(ASE.Compiler.CodePurpose.SKELETON, self.baseDir)
      try:
        map(lambda tree: tree.accept(output), source[language])
        for f in output.getFiles():
          self.logPrint('Created '+str(language)+' file '+str(f))
      except ASE.BaseException.Exception, e:
        print e.getMessage()
    del source
    return
  
  def run(self):
    self.setup()
    self.logPrint('Making a P1 element')
    self.outputElementSource(self.shape, 1)
    return

if __name__ == '__main__':
  P1().run()
