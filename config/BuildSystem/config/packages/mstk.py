import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.download         = ['http://software.lanl.gov/ascem/tpls/mstk-2.23.tgz']
    self.downloadfilename = 'mstk'
    self.includes         = ['MSTK.h']
    self.functions        = []
    self.cxx              = 1
    self.requirescxx11    = 0
    self.downloadonWindows= 0
    self.hastests         = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.deps            = [self.mpi]
    return


