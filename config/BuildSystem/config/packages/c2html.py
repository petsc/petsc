import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = 'v0.9.4-p3'
    self.download          = ['git://https://gitlab.com/petsc/pkg-c2html.git' ,
                              'https://gitlab.com/petsc/pkg-c2html/-/archive/'+self.gitcommit+'/pkg-c2html-'+self.gitcommit+'.tar.gz']
    self.downloaddirnames  = ['pkg-c2html']
    self.downloadonWindows = 1
    self.publicInstall     = 0  # always install in PETSC_DIR/PETSC_ARCH (not --prefix) since this is not used by users
    self.parallelMake      = 0
    self.lookforbydefault  = 1
    self.executablename    = 'c2html'
    self.skippackagelibincludedirs = 1

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('C2HTML', '-download-c2html-cc=<prog>',                     nargs.Arg(None, None, 'C compiler for c2html'))
    help.addArgument('C2HTML', '-with-c2html-exec=<executable>',                 nargs.Arg(None, None, 'c2html executable to look for'))
    return

  def formGNUConfigureArgs(self):
    '''Does not use the standard arguments at all since this does not use the MPI compilers etc
       Sowing will chose its own compilers if they are not provided explicitly here'''
    args = ['--prefix='+self.installDir]
    if 'download-c2html-cc' in self.argDB and self.argDB['download-c2html-cc']:
      args.append('CC="'+self.argDB['download-c2html-cc']+'"')
    return args

  def locateC2html(self):
    '''Determine location of c2html executable'''
    if 'with-c2html-exec' in self.argDB:
      self.log.write('Looking for specified C2html executable '+self.argDB['with-c2html-exec']+'\n')
      self.getExecutable(self.argDB['with-c2html-exec'], getFullPath=1, resultName='c2html')
    else:
      self.log.write('Looking for default C2html executable\n')
      self.getExecutable('c2html', getFullPath=1, resultName='c2html')
    return

  def Install(self):
    # check if flex or lex are in PATH
    if not hasattr(self.programs, 'flex') and not hasattr(self.programs, 'lex'):
      self.programs.getExecutable('flex', getFullPath = 1)
      self.programs.getExecutable('lex')
    if not hasattr(self.programs, 'flex') and not hasattr(self.programs, 'lex'):
      raise RuntimeError('Cannot build c2html. It requires either "flex" or "lex" in PATH. Please install flex and retry.\nOr disable c2html with --with-c2html=0')
    return config.package.GNUPackage.Install(self)

  def configure(self):
    '''Locate c2html and download it if requested'''
    if self.argDB['download-c2html']:
      self.log.write('Building c2html\n')
      config.package.GNUPackage.configure(self)
      self.getExecutable('c2html',    path=os.path.join(self.installDir,'bin'), getFullPath = 1, resultName='c2html')
    elif (not self.argDB['with-c2html']  == 0 and not self.argDB['with-c2html']  == 'no') or 'with-c2html-exec' in self.argDB:
      self.executeTest(self.locateC2html)
    else:
      self.log.write('Not checking for C2html\n')
    if hasattr(self, 'c2html'): self.found = 1
    return
