import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.libraries    = self.framework.require('config.libraries', self)
    self.include      = []
    self.lib          = None
    return

  def __str__(self):
    return ''

  def setupHelp(self, help):
    import nargs

    help.addArgument('ASE', '-aseVersion', nargs.ArgInt(None, 2, 'Specify the ASE version'))
    return

  def configureASEVersion(self):
    self.version = self.argDB['aseVersion']
    if self.version == 1:
      self.baseName = 'sidl'
    elif self.version == 2:
      self.baseName = 'ase'
    return

  def configureASELibraries(self):
    import os

    self.dir = self.argDB['ASE_DIR']
    self.lib = [os.path.join(self.argDB['ASE_DIR'], 'lib', 'lib-python-'+self.baseName+'.so')]
    if not os.path.isdir(self.argDB['ASE_DIR']):
      raise RuntimeError('Invalid ASE directory: '+str(self.argDB['ASE_DIR']))
    if not os.path.samefile(os.getcwd(), self.argDB['ASE_DIR']):
      for lib in self.lib:
        if not os.path.isfile(lib):
          raise RuntimeError('Invalid ASE library: '+str(lib))
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
    self.executeTest(self.configureASEVersion)
    self.executeTest(self.configureASELibraries)
    self.setOutput()
    return
