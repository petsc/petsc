import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.includes     = ['mkl_pardiso.h']
    self.functions    = ['pardisoinit']
    self.liblist      = [[]] # use MKL detected by BlasLapack.py
    self.double       = 0
    self.requires32bitint = 0
    self.complex      = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return
