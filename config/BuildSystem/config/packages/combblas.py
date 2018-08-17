import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.download          = ['http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/CombBLAS_beta_16_1.tgz']
    self.functions         = ['ParMETIS_V3_PartKway']
    self.includes          = ['parmetis.h']
    self.liblist           = [['libparmetis.a']]
    self.hastests          = 1
    self.cxx               = 1
    self.requirescxx11     = 1
    self.downloaddirnames  = ['CombBLAS']


  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.mpi           = framework.require('config.packages.MPI',self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.deps          = [self.mpi, self.mathlib]



