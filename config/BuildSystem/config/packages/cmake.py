import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download          = ['http://www.cmake.org/files/v3.0/cmake-3.0.1.tar.gz']
    self.downloadonWindows = 1
    self.lookforbydefault  = 1
    return

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('CMAKE', '-download-cmake-cc=<prog>',                   nargs.Arg(None, None, 'C compiler for Cmake configure'))
    help.addArgument('CMAKE', '-download-cmake-configure-options=<options>', nargs.Arg(None, None, 'Additional options for Cmake configure'))
    help.addArgument('CMAKE', '-with-cmake-exec=<executable>',                nargs.Arg(None, None, 'CMake executable to look for'))
    return

  def formGNUConfigureArgs(self):
    '''Does not use the standard arguments at all since this does not use the MPI compilers etc
       Cmake will chose its own compilers if they are not provided explicitly here'''
    args = ['--prefix='+self.confDir]
    if 'download-cmake-cc' in self.framework.argDB and self.framework.argDB['download-cmake-cc']:
      args.append('CC="'+self.framework.argDB['download-cmake-cc']+'"')
    if 'download-cmake-configure-options' in self.framework.argDB and self.framework.argDB['download-cmake-configure-options']:
      args.append(self.framework.argDB['download-cmake-configure-options'])
    return args

  def locateCMake(self):
    if 'with-cmake-exec' in self.framework.argDB:
      self.framework.log.write('Looking for specified CMake executable '+self.framework.argDB['with-cmake-exec']+'\n')
      self.getExecutable(self.framework.argDB['with-cmake-exec'], getFullPath=1, resultName='cmake')
    else:
      self.framework.log.write('Looking for default CMake executable\n')
      self.getExecutable('cmake', getFullPath=1, resultName='cmake')
    return

  def alternateConfigureLibrary(self):
    self.checkDownload(1)

  def configure(self):
    '''Locate cmake and download it if requested'''
    if self.framework.argDB['download-cmake']:
      self.framework.log.write('Building CMake\n')
      config.package.GNUPackage.configure(self)
      self.getExecutable('cmake',    path=os.path.join(self.installDir,'bin'), getFullPath = 1)
    elif (not self.framework.argDB['with-cmake']  == 0 and not self.framework.argDB['with-cmake']  == 'no') or 'with-cmake-exec' in self.framework.argDB:
      self.executeTest(self.locateCMake)
    else:
      self.framework.log.write('Not checking for CMake\n')
    if hasattr(self, 'cmake'): self.found = 1
    return
