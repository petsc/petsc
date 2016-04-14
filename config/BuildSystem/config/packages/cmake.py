import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download          = ['http://www.cmake.org/files/v3.0/cmake-3.0.1.tar.gz',
                              'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/cmake-3.0.1.tar.gz']
    self.downloadonWindows = 1
    self.lookforbydefault  = 1
    self.publicInstall     = 0  # always install in PETSC_DIR/PETSC_ARCH (not --prefix) since this is not used by users
    return

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('CMAKE', '-download-cmake-cc=<prog>',                   nargs.Arg(None, None, 'C compiler for Cmake configure'))
    help.addArgument('CMAKE', '-download-cmake-configure-options=<options>', nargs.Arg(None, None, 'Additional options for Cmake configure'))
    help.addArgument('CMAKE', '-with-cmake-exec=<executable>',                nargs.Arg(None, None, 'CMake executable to look for'))
    help.addArgument('CMAKE', '-with-ctest-exec=<executable>',                nargs.Arg(None, None, 'Ctest executable to look for'))
    return

  def formGNUConfigureArgs(self):
    '''Does not use the standard arguments at all since this does not use the MPI compilers etc
       Cmake will chose its own compilers if they are not provided explicitly here'''
    args = ['--prefix='+self.confDir]
    if 'download-cmake-cc' in self.argDB and self.argDB['download-cmake-cc']:
      args.append('CC="'+self.argDB['download-cmake-cc']+'"')
    if 'download-cmake-configure-options' in self.argDB and self.argDB['download-cmake-configure-options']:
      args.append(self.argDB['download-cmake-configure-options'])
    return args

  def locateCMake(self):
    if 'with-cmake-exec' in self.argDB:
      self.log.write('Looking for specified CMake executable '+self.argDB['with-cmake-exec']+'\n')
      self.getExecutable(self.argDB['with-cmake-exec'], getFullPath=1, resultName='cmake')
    else:
      self.log.write('Looking for default CMake executable\n')
      self.getExecutable('cmake', getFullPath=1, resultName='cmake')
    if 'with-ctest-exec' in self.argDB:
      self.log.write('Looking for specified Ctest executable '+self.argDB['with-ctest-exec']+'\n')
      self.getExecutable(self.argDB['with-ctest-exec'], getFullPath=1, resultName='ctest')
    else:
      self.log.write('Looking for default CTest executable\n')
      self.getExecutable('ctest', getFullPath=1, resultName='ctest')
    return

  def alternateConfigureLibrary(self):
    self.checkDownload()

  def configure(self):
    '''Locate cmake and download it if requested'''
    if self.argDB['download-cmake']:
      self.log.write('Building CMake\n')
      config.package.GNUPackage.configure(self)
      self.log.write('Looking for Cmake in '+os.path.join(self.installDir,'bin')+'\n')
      self.getExecutable('cmake',    path=os.path.join(self.installDir,'bin'), getFullPath = 1)
      self.getExecutable('ctest',    path=os.path.join(self.installDir,'bin'), getFullPath = 1)
    elif (not self.argDB['with-cmake']  == 0 and not self.argDB['with-cmake']  == 'no') or 'with-cmake-exec' in self.argDB:
      self.executeTest(self.locateCMake)
    else:
      self.log.write('Not checking for CMake\n')
    if hasattr(self, 'cmake'): self.found = 1
    return
