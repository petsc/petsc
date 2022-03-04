import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '5bad7487f496c811192334640ce4d3fc5f88144b' # main mar-18-2022
    self.download         = ['git://https://github.com/Reference-ScaLAPACK/scalapack','https://github.com/Reference-ScaLAPACK/scalapack/archive/'+self.gitcommit+'.tar.gz']
    self.includes         = []
    self.liblist          = [['libscalapack.a'],
                             ['libmkl_scalapack_lp64.a','libmkl_blacs_intelmpi_lp64.a'],
                             ['libmkl_scalapack_lp64.a','libmkl_blacs_mpich_lp64.a'],
                             ['libmkl_scalapack_lp64.a','libmkl_blacs_sgimpt_lp64.a'],
                             ['libmkl_scalapack_lp64.a','libmkl_blacs_openmpi_lp64.a']]
    self.functions        = ['pssytrd']
    self.functionsFortran = 1
    self.buildLanguages   = ['FC']
    self.precisions       = ['single','double']
    self.downloadonWindows= 1
    self.makerulename     = 'scalapack'
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.flibs      = framework.require('config.packages.flibs',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.deps       = [self.mpi, self.blasLapack, self.flibs]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DLAPACK_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DSCALAPACK_BUILD_TESTS=OFF')
    return args

def getSearchDirectories(self):
  '''Generate list of possible locations of ScaLAPACK'''
  yield ''
  if os.getenv('MKLROOT'): yield os.getenv('MKLROOT')
