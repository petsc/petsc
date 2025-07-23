import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '2.2.2'
    self.gitcommit        = '6f56981cb0cabffd8c72c7d1016146c4b8e276dc' # master (includes fix for C std) jul-16-2025
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
    self.minCmakeVersion  = (3,26,0)
    self.libDirs          = ['lib',os.path.join('lib','intel64')]
    self.requirekandr     = 1
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
  if 'with-'+self.package+'-dir' in self.argDB:
    d = self.argDB['with-'+self.package+'-dir']
    for libdir in self.libDirs:
      yield os.path.join(d,libdir)
  if os.getenv('MKLROOT'):
    for libdir in self.libDirs:
      yield os.path.join(os.getenv('MKLROOT'),libdir)
