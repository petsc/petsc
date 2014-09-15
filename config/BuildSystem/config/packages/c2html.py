import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download          = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/c2html.tar.gz']
    self.complex           = 1
    self.double            = 0
    self.requires32bitint  = 0
    self.downloadonWindows = 1
    self.publicInstall     = 0  # always install in PETSC_DIR/PETSC_ARCH (not --prefix) since this is not used by users
    self.parallelMake      = 0  

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('C2html', '-download-c2html-cc=<prog>',                     nargs.Arg(None, None, 'C compiler for c2html'))
    help.addArgument('C2html', '-download-c2html-configure-options=<options>',   nargs.Arg(None, None, 'additional options for c2html'))
    return

  def formGNUConfigureArgs(self):
    '''Does not use the standard arguments at all since this does not use the MPI compilers etc
       Sowing will chose its own compilers if they are not provided explicitly here'''
    args = ['--prefix='+self.confDir]
    if 'download-c2html-cc' in self.framework.argDB and self.framework.argDB['download-c2html-cc']:
      args.append('CC="'+self.framework.argDB['download-c2html-cc']+'"')
    if 'download-c2html-configure-options' in self.framework.argDB and self.framework.argDB['download-c2html-configure-options']:
      args.append(self.framework.argDB['download-c2html-configure-options'])
    return args

  def Install(self):
    # check if flex or lex are in PATH
    self.getExecutable('flex')
    self.getExecutable('lex')
    if not hasattr(self, 'flex') and not hasattr(self, 'lex'):
      raise RuntimeError('Cannot build c2html. It requires either "flex" or "lex" in PATH. Please install flex and retry.\nOr disable c2html with --with-c2html=0')
    return config.package.GNUPackage.Install(self)

