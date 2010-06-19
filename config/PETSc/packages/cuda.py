import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions = ['cublasInit']
    self.includes  = ['cublas.h']
    self.liblist   = [['libcublas.a']]
    self.double    = 0   # 1 means requires double precision 
    self.cxx       = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.headers        = framework.require('config.headers',self)

  def getSearchDirectories(self):
    yield os.path.join('/usr','local','cuda')

  def configureLibrary(self):
    PETSc.package.NewPackage.configureLibrary(self)
    self.setCompilers.pushLanguage('C++')
#    if not self.headers.checkInclude([os.path.join('/usr','local','cuda')],[os.path.join('thrust','advance.h')]):
#       raise RuntimeError('Cannot find thrust include files') 
#    if not self.headers.checkInclude([os.path.join('/usr','local','cuda')],[os.path.join('cusp','monitor.h')]):
#       raise RuntimeError('Cannot find cusp include files') 
    self.setCompilers.popLanguage()
    self.include = self.include+[os.path.join('/usr','local','cuda')]

    
# add checks that it is proper version o Cuda
