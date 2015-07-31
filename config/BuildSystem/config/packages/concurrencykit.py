import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download  = ['http://concurrencykit.org/releases/ck-0.4.5.tar.gz']
    self.functions = []
    self.includes  = ['ck_spinlock.h']
    self.liblist   = [['libck.a']]
    self.downloadonWindows = 0
    self.double            = 1
    self.complex           = 0
    self.downloadfilename  = 'ck'

  # this should not be needed but configure crashes if it is
  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.deps           = [self.mpi,self.blasLapack]

