import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download          = ['http://www.cmake.org/files/v2.8/cmake-2.8.12.2.tar.gz']
    self.downloadonWindows = 1

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('Cmake', '-download-cmake-cc=<prog>',                     nargs.Arg(None, None, 'C compiler for cmake configure'))
    help.addArgument('Cmake', '-download-cmake-configure-options=<options>',   nargs.Arg(None, None, 'additional options for cmake configure'))
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

  def alternateConfigureLibrary(self):
    self.checkDownload(1)

  def configure(self):
    '''Locate cmake and download it if requested'''
    if self.framework.argDB['download-cmake']:
      config.package.GNUPackage.configure(self)
      self.getExecutable('cmake',    path=os.path.join(self.installDir,'bin'), getFullPath = 1)
    elif self.framework.argDB['with-cmake']:
      if self.framework.argDB['with-cmake']  == 1 or self.framework.argDB['with-cmake']  == 'yes':
        self.getExecutable('cmake', getFullPath = 1,resultName='cmake')
      elif not self.framework.argDB['with-cmake']  == 0 and not self.framework.argDB['with-cmake']  == 'no':
        self.getExecutable(self.framework.argDB['with-cmake'], getFullPath = 1,resultName='cmake')
    if not hasattr(self, 'cmake'):
      self.found = 1
    return
