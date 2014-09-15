import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions        = ['PAMI_Client_create']
    self.includes         = ['pami.h']
    self.liblist          = [['libpami.a']]
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.deps = [self.mpi]
    return

  def getSearchDirectories(self):
    yield ''
    yield '/bgsys/drivers/V1R1M0/ppc64/comm/sys'
