import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.includes     = ['mkl.h']
    self.functions    = ['pardisoinit']
    self.liblist      = [[]] # use MKL detected by BlasLapack.py
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
        raise RuntimeError('MKL_CParadiso requires Intel MKL. Please rerun configure using --with-blas-lapack-dir=LOCATION_OF_INTEL_MKL')
    return
