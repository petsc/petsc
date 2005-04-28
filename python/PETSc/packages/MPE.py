import config.base

import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.argDB        = framework.argDB
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.mpi = framework.require('PETSc.packages.MPI', self)
    return

  def configure(self):
    return
