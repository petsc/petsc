import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.includes     = ['mkl.h']
    self.functions    = ['pardisoinit']
    self.liblist      = [[]] # use MKL detected by BlasLapack.py
    self.precisions   = ['single','double']
    self.hastests     = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      if not self.blasLapack.mkl:
        raise RuntimeError('MKL_Pardiso requires Intel MKL. Please rerun configure using --with-blaslapack-dir=LOCATION_OF_INTEL_MKL')
    return
