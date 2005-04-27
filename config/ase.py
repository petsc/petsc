import config.base

import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.include      = []
    self.lib          = None
    self.baseName     = 'ase'
    return

  def __str__(self):
    return ''

  def setupHelp(self, help):
    import nargs

    help.addArgument('ASE', '-ase-dir', nargs.ArgDir(None, None, 'Specify the ASE Runtime directory'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers', self)
    self.libraries    = framework.require('config.libraries', self)
    return

  def checkASEDir(self, dir):
    '''Try to determine if this is a valid Runtime directory'''
    if not os.path.isdir(self.dir):
      return 0
    if not os.path.isdir(os.path.join(self.dir, 'server-python-ase')):
      return 0
    if not os.path.isfile(os.path.join(self.dir, 'make.py')):
      return 0
    if not os.path.isfile(os.path.join(self.dir, 'make.py')):
      return 0
    return 1

  def configureASELibraries(self):
    self.dir = ''
    if 'ase-dir' in self.argDB:
      self.dir = self.argDB['ase-dir']
    else:
      aseDir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'ase', 'Runtime')
      if os.path.isdir(aseDir):
        self.dir = aseDir
    if not self.checkASEDir(self.dir):
      raise RuntimeError('Invalid ASE directory: '+str(self.dir))
    self.logPrint('ASE directory is '+str(self.dir))
    self.lib = [os.path.join(self.dir, 'lib', 'lib-python-ase.'+self.setCompilers.sharedLibraryExt)]
    if not os.path.samefile(os.getcwd(), self.dir):
      for lib in self.lib:
        if not os.path.isfile(lib):
          raise RuntimeError('Invalid ASE library: '+str(lib))
    self.logPrint('ASE libraries are '+str(self.lib))
    return

  def configureScandal(self):
    '''Set SCANDAL_DIR to the scandal directory for now'''
    self.scandalDir = os.path.join(os.path.dirname(self.dir), 'Compiler', 'driver', 'python')
    if os.path.isdir(self.scandalDir):
      self.argDB['SCANDAL_DIR'] = self.scandalDir
    return

  def setOutput(self):
    '''Add defines and substitutions
       - ASE_LIB is the command line arguments for the link
       - ASE_LIBRARY is the list of ASE libraries'''
    if self.lib:
      self.addSubstitution('ASE_LIB',     ' '.join(map(self.libraries.getLibArgument, self.lib)))
      self.addSubstitution('ASE_LIBRARY', self.lib)
    else:
      self.addSubstitution('ASE_LIB',     '')
      self.addSubstitution('ASE_LIBRARY', '')
    return

  def configure(self):
    self.executeTest(self.configureASELibraries)
    self.executeTest(self.configureScandal)
    self.executeTest(self.setOutput)
    return
