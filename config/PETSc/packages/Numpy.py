import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.includes         = ['']
    self.includedir       = ''
    self.libdir           = ''
    self.complex          = 0   # 0 means cannot use complex
    self.cxx              = 0   # 1 means requires C++
    self.fc               = 0   # 1 means requires fortran
    self.double           = 1   # 1 means requires double precision
    self.requires32bitint = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.petscdir = self.framework.require('PETSc.utilities.petscdir',self)
    return

  def configureLibrary(self):
    try:
      import numpy
    except:
      raise RuntimeError('Could not find numpy, either fix PYTHONPATH and rerun or install it')
    return
