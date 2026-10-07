import os

import config.package

class Configure(config.package.MesonPackage):
  def __init__(self, framework):
    config.package.MesonPackage.__init__(self, framework)
    self.version          = '3.1.1'
    self.gitcommit        = '979967c923241c136c35f17630cfdcf614e6df17' # 3.1.1
    self.download         = ['git://https://github.com/ralna/SIFDecode.git', 'https://github.com/ralna/SIFDecode/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames = ['SIFDecode']
    self.buildLanguages   = ['FC']
    self.executablename   = 'sifdecoder'
    self.linkedbypetsc    = 0
    self.skippackagelibincludedirs = 1

  def setupHelp(self, help):
    import nargs
    config.package.MesonPackage.setupHelp(self, help)
    help.addArgument('SIFDECODE', '-with-sifdecode-exec=<executable>', nargs.Arg(None, None, 'SIFDecode standalone decoder executable'))

  def configureLibrary(self):
    directory = self.checkDownload()
    if directory:
      executable = os.path.join(directory, 'bin', 'sifdecoder')
    elif 'with-sifdecode-exec' in self.argDB:
      executable = self.argDB['with-sifdecode-exec']
    elif 'with-sifdecode-dir' in self.argDB:
      executable = os.path.join(self.argDB['with-sifdecode-dir'], 'bin', 'sifdecoder')
    else:
      executable = 'sifdecoder'
    self.getExecutable(executable, getFullPath=1, resultName='sifdecoder')
    if not hasattr(self, 'sifdecoder'):
      raise RuntimeError('SIFDecode was not found. Use --download-sifdecode or --with-sifdecode-exec.')
    self.executeShellCommand([self.sifdecoder, '-h'], timeout=30, log=self.log)
    self.found = 1
    self.directory = directory or os.path.dirname(os.path.dirname(self.sifdecoder))
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

  def configure(self):
    if 'with-sifdecode-exec' in self.argDB:
      self.argDB['with-sifdecode'] = 1
    config.package.MesonPackage.configure(self)
