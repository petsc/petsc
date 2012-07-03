import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.lookforbydefault  = 0
    self.functions         = 0
    self.includes          = 0
    self.liblist           = 0
    self.complex           = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.pthread = framework.require('PETSc.packages.pthread',self)
    self.deps    = []
    return

  def configureLibrary(self):
    if self.pthread.found:
        self.addDefine('USE_PTHREAD',1)
        self.addDefine('USE_SERVER',1)    
    
