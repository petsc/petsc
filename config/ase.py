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

  def configureASELibraries(self):
    import os

    self.lib = [os.path.join(self.argDB['ASE_DIR'], 'lib', 'libase.so')]
    if not os.path.isdir(self.argDB['ASE_DIR']):
      raise RuntimeError('Invalid ASE directory: '+str(self.argDB['ASE_DIR']))
    if not os.getcwd() == self.argDB['ASE_DIR']:
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
    self.executeTest(self.configureASELibraries)
    self.setOutput()
    return
