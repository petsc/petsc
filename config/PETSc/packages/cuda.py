import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions = ['cublasInit']
    self.includes  = ['cublas.h']
    self.liblist   = [['libcublas.a']]
    return

  def getSearchDirectories(self):
    yield os.path.join('/usr','local','cuda')
