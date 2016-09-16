import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'v5.1.0-p3'
    self.download          = ['git://https://bitbucket.org/petsc/pkg-metis.git','http://ftp.mcs.anl.gov/pub/petsc/externalpackages/metis-5.1.0-p3.tar.gz']
    self.functions         = ['METIS_PartGraphKway']
    self.includes          = ['metis.h']
    self.liblist           = [['libmetis.a'],['libmetis.a','libexecinfo.a']]
    self.needsMath         = 1
    self.hastests          = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    return

  def formCMakeConfigureArgs(self):
    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build METIS\nSuggest adding --download-cmake to ./configure arguments')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DGKLIB_PATH=../GKlib')
    # force metis/parmetis to use a portable random number generator that will produce the same partitioning results on all systems
    args.append('-DGKRAND=1')
    if self.checkSharedLibrariesEnabled():
      args.append('-DSHARED=1')
    if self.compilerFlags.debugging:
      args.append('-DDEBUG=1')
    if self.getDefaultIndexSize() == 64:
      args.append('-DMETIS_USE_LONGINDEX=1')
    if self.getDefaultPrecision() == 'double':
      args.append('-DMETIS_USE_DOUBLEPRECISION=1')
    elif self.getDefaultPrecision() == 'quad':
      raise RuntimeError('METIS cannot be built with quad precision')
    return args
