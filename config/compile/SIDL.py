import script

class Compiler(script.Script):
  '''The SIDL compiler'''
  def __init__(self, argDB):
    script.Script.__init__(self, argDB = argDB)
    self.language        = 'SIDL'
    self.sourceExtension = '.sidl'
    self.clients         = []
    self.clientDirs      = {}
    self.servers         = []
    self.serverDirs      = {}
    self.includes        = []
    return

  def __getstate__(self):
    '''We do not want to pickle Scandal'''
    d = script.Script.__getstate__(self)
    if 'scandal' in d:
      del d['scandal']
    return d

  def createClient(self, outputFiles):
    self.scandal.clients    = self.clients
    self.scandal.clientDirs = self.clientDirs
    self.scandal.servers    = []
    self.scandal.serverDirs = {}
    self.scandal.run()
    for lang in self.scandal.outputFiles:
      outputFiles['Client '+lang] = self.scandal.outputFiles[lang]
    return

  def createServer(self, outputFiles):
    self.scandal.clients    = []
    self.scandal.clientDirs = {}
    self.scandal.servers    = self.servers
    self.scandal.serverDirs = self.serverDirs
    self.scandal.run()
    for lang in self.scandal.outputFiles:
      outputFiles['Server '+lang] = self.scandal.outputFiles[lang]
    return

  def __call__(self, source, target = None):
    '''This will compile the SIDL source'''
    outputFiles             = {}
    self.scandal.includes   = self.includes
    self.scandal.targets    = source
    self.createServer(outputFiles)
    self.createClient(outputFiles)
    return ('', '', 0, outputFiles)

  def checkSetup(self):
    '''Check that this module has been specified. We assume that configure has checked its viability.'''
    import os

    if not hasattr(self, 'scandal'):
      self.scandal = self.getModule(self.argDB['SCANDAL_DIR'], 'scandal').Scandal(argDB = self.argDB)
      self.scandal.setup()
    return self.scandal

  def getTarget(self, source):
    '''Returns the default target for the given source file, or None'''
    return None
