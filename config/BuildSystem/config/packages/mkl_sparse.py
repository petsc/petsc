import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.includes         = ['mkl.h','mkl_spblas.h']
    self.functions        = ['mkl_dcsrmv']
    self.liblist          = [[]] # use MKL detected by BlasLapack.py
    self.precisions       = ['single','double']
    self.lookforbydefault = 1
    self.requires32bitint = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return

  def configureLibrary(self):
    if not self.blasLapack.mkl: return
    config.package.Package.configureLibrary(self)
    self.usesopenmp = self.blasLapack.usesopenmp
