import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version           = '3.8.2'
    self.download          = ['https://ftp.gnu.org/gnu/bison/bison-'+self.version+'.tar.gz',
                              'http://mirrors.kernel.org/gnu/bison/bison-'+self.version+'.tar.gz']
    self.lookforbydefault  = 1
    self.haveBison3plus    = 0
    self.publicInstall     = 0 # always install in PETSC_DIR/PETSC_ARCH (not --prefix) since this is not used by users
    self.executablename    = 'bison'

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('BISON', '-download-bison-cc=<prog>',                     nargs.Arg(None, None, 'C compiler for Bison'))
    help.addArgument('BISON', '-with-bison-exec=<executable>',                 nargs.Arg(None, None, 'Bison executable to look for'))
    return

  def formGNUConfigureArgs(self):
    '''Does not use the standard arguments at all since this does not use the MPI compilers etc
       Bison will chose its own compilers if they are not provided explicitly here'''
    args = ['--prefix='+self.installDir]
    if 'download-bison-cc' in self.argDB and self.argDB['download-bison-cc']:
      args.append('CC="'+self.argDB['download-bison-cc']+'"')
    return args

  def locateBison(self):
    if 'with-bison-exec' in self.argDB:
      self.log.write('Looking for specified Bison executable '+self.argDB['with-bison-exec']+'\n')
      self.getExecutable(self.argDB['with-bison-exec'], getFullPath=1, resultName='bison')
    else:
      self.log.write('Looking for default Bison executable\n')
      self.getExecutable('bison', getFullPath=1)
    return

  def alternateConfigureLibrary(self):
    self.checkDownload()

  def configure(self):
    '''Locate Bison and download it if requested'''
    if self.argDB['download-bison']:
      # check if flex or lex are in PATH
      self.getExecutable('flex')
      self.getExecutable('lex')
      if not hasattr(self, 'flex') and not hasattr(self, 'lex'):
        raise RuntimeError('Cannot build Bison. It requires either "flex" or "lex" in PATH. Please install flex and retry.\nOr disable Bison with --with-bison=0')
      self.log.write('Building Bison\n')
      config.package.GNUPackage.configure(self)
      self.log.write('Looking for Bison in '+os.path.join(self.installDir,'bin')+'\n')
      self.getExecutable('bison', path = os.path.join(self.installDir,'bin'), getFullPath = 1)
    elif (not self.argDB['with-bison'] == 0 and not self.argDB['with-bison']  == 'no') or 'with-bison-exec' in self.argDB:
      self.executeTest(self.locateBison)
    else:
      self.log.write('Not checking for Bison\n')
    if hasattr(self, 'bison'):
      self.found = 1
      try:
        import re
        (output, error, status) = config.base.Configure.executeShellCommand(self.bison+' --version', log = self.log)
        gver = re.compile('bison \(GNU Bison\) ([0-9]+).([0-9]+)').match(output)
        if not status and gver:
          foundversion = tuple(map(int,gver.groups()))
          self.foundversion = ".".join(map(str,foundversion))
          if foundversion[0] >= 3:
            self.haveBison3plus = 1
          else:
            self.logPrintBox('***** WARNING: You have a version of GNU Bison older than 3.0. It will work,\n\
but may not be supported by all external packages. You can install the \n\
latest GNU Bison with your package manager, such as Brew or MacPorts, or use\n\
the --download-bison option to get the latest GNU Bison *****')
      except RuntimeError as e:
        self.log.write('Bison check failed: '+str(e)+'\n')
    return
