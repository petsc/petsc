import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions        = ['PAMI_Client_create']
    self.includes         = ['pami.h']
    self.liblist          = [['libpami.a']]
    self.complex          = 1
    self.double           = 0
    self.requires32bitint = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = [self.mpi]
    return

  def getSearchDirectories(self):
    yield ''
    yield '/bgsys/drivers/V1R1M0/ppc64/comm/sys'
