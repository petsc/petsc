import script

class Compiler(script.Script):
  '''The SIDL compiler'''
  def __init__(self, argDB):
    import re

    script.Script.__init__(self, argDB = argDB)
    self.language        = 'SIDL'
    self.sourceExtension = '.sidl'
    self.implRE          = re.compile(r'^((.*)_impl\.(c|h|py)|__init__\.py)$')
    self.scandalDir      = None
    self.clients         = []
    self.clientDirs      = {}
    self.servers         = []
    self.serverDirs      = {}
    self.includes        = []
    self.includeDirectories = {}
    self.systemIncludeDirectories = {}
    self.versionControl  = None
    self.useShell        = 1
    self.disableOutput   = 0
    return

  def __getstate__(self):
    '''We do not want to pickle Scandal'''
    d = script.Script.__getstate__(self)
    if 'scandal' in d:
      del d['scandal']
    return d

  def createClient(self, source, outputFiles):
    self.scandal.clients    = self.clients
    self.scandal.clientDirs = self.clientDirs
    self.scandal.servers    = []
    self.scandal.serverDirs = {}
    self.scandal.outputSIDLFiles = not self.disableOutput
    if 'baseDirectory' in self.argDB:
      self.scandal.baseDirectory = self.argDB['baseDirectory']
    self.scandal.run()
    for lang in self.scandal.outputFiles:
      outputFiles['Client '+lang] = self.scandal.outputFiles[lang]
    return

  def filterShellOutput(self, output):
    output = [l.strip() for l in output.split('\n')]
    output = filter(lambda l: not l.startswith('Mapping'), output)
    output = filter(lambda l: not l.startswith('Searching'), output)
    output = filter(lambda l: not l.startswith('Creating'), output)
    output = filter(lambda l: not l.startswith('Parsing'), output)
    output = filter(lambda l: not l.startswith('Failed to'), output)
    output = filter(lambda l: not l.startswith('Failed to'), output)
    output = filter(lambda l: not l.startswith('scandal:'), output)
    self.logPrint('Got scandal output: '+str(output))
    return ''.join(output)

  def createClientShell(self, source, outputFiles):
    import os
    import sys
    from sets import Set

    for client in self.clients:
      cmd = [sys.executable, os.path.join(self.scandalDir, 'scandal.py')]
      cmd.append('--client='+client)
      cmd.append('--clientDirs={'+client+':'+self.clientDirs[client]+'}')
      cmd.append('--includes=['+','.join(self.includes)+']')
      cmd.append('--ior=0')
      cmd.append('--outputSIDLFiles='+str(not self.disableOutput))
      cmd.append('--outputFiles')
      cmd.append('--logAppend')
      if 'baseDirectory' in self.argDB:
        cmd.append('--baseDirectory='+self.argDB['baseDirectory'])
      cmd.extend(source)
      (output, error, status) = self.executeShellCommand(' '.join(cmd), timeout = None)
      scandalOutputFiles = eval(self.filterShellOutput(output))
      for lang in scandalOutputFiles:
        outputFiles['Client '+lang] = scandalOutputFiles[lang]
      cmd = [os.path.join(self.scandalDir, 'scandal.py')]
      cmd.append('--ior=client')
      cmd.append('--clientDirs={'+client+':'+self.clientDirs[client]+'}')
      cmd.append('--includes=['+','.join(self.includes)+']')
      cmd.append('--outputSIDLFiles='+str(not self.disableOutput))
      cmd.append('--outputFiles')
      cmd.append('--logAppend')
      if 'baseDirectory' in self.argDB:
        cmd.append('--baseDirectory='+self.argDB['baseDirectory'])
      cmd.extend(source)
      (output, error, status) = self.executeShellCommand(' '.join(cmd), timeout = None)
      scandalOutputFiles = eval(self.filterShellOutput(output))
      for lang in scandalOutputFiles:
        outputFiles['Client '+lang] = scandalOutputFiles[lang]
    return

  def editServer(self, serverDirs):
    import os

    vc = self.versionControl
    for serverDir in serverDirs.values():
      for root, dirs, files in os.walk(serverDir):
        if os.path.basename(root) == 'SCCS':
          continue
        vc.edit(vc.getClosedFiles([os.path.join(root, f) for f in filter(lambda a: self.implRE.match(a), files)]))
    return

  def checkinServer(self, serverDirs):
    import os

    vc        = self.versionControl
    added     = 0
    reverted  = 0
    committed = 0
    for serverDir in serverDirs.values():
      for root, dirs, files in os.walk(serverDir):
        if os.path.basename(root) == 'SCCS':
          continue
        try:
          implFiles = filter(lambda a: self.implRE.match(a), files)
          added     = added or vc.add(vc.getNewFiles([os.path.join(root, f) for f in implFiles]))
          reverted  = reverted or vc.revert(vc.getUnchangedFiles([os.path.join(root, f) for f in implFiles]))
          committed = committed or vc.commit(vc.getChangedFiles([os.path.join(root, f) for f in implFiles]))
        except RuntimeError, e:
          self.logPrint('ERROR: Checking in server: '+str(e))
    if added or committed:
      try:
        vc.changeSet()
      except RuntimeError, e:
        self.logPrint('ERROR: Checking in server: '+str(e))
    return

  def createServer(self, source, outputFiles):
    self.scandal.clients    = []
    self.scandal.clientDirs = {}
    self.scandal.servers    = self.servers
    self.scandal.serverDirs = self.serverDirs
    self.scandal.includeDirectories = self.includeDirectories
    self.scandal.systemIncludeDirectories = self.systemIncludeDirectories
    self.scandal.outputSIDLFiles = not self.disableOutput
    if 'baseDirectory' in self.argDB:
      self.scandal.baseDirectory = self.argDB['baseDirectory']
    self.editServer(self.serverDirs)
    self.scandal.run()
    for lang in self.scandal.outputFiles:
      outputFiles['Server '+lang] = self.scandal.outputFiles[lang]
    self.checkinServer(self.serverDirs)
    return

  def createServerShell(self, source, outputFiles):
    import os
    import sys
    from sets import Set

    self.editServer(self.serverDirs)
    for server in self.servers:
      cmd = [sys.executable, os.path.join(self.scandalDir, 'scandal.py')]
      cmd.append('--server='+server)
      cmd.append('--serverDirs={'+server+':'+self.serverDirs[server]+'}')
      cmd.append('--includes=['+','.join(self.includes)+']')
      if server in self.includeDirectories:
        cmd.append('--includeDirs="{'+server+':['+','.join(self.includeDirectories[server])+']}"')
      if server in self.systemIncludeDirectories:
        cmd.append('--systemIncludeDirs="{'+server+':['+','.join(self.systemIncludeDirectories[server])+']}"')
      cmd.append('--ior=0')
      cmd.append('--outputSIDLFiles='+str(not self.disableOutput))
      cmd.append('--outputFiles')
      cmd.append('--logAppend')
      if 'baseDirectory' in self.argDB:
        cmd.append('--baseDirectory='+self.argDB['baseDirectory'])
      cmd.extend(source)
      (output, error, status) = self.executeShellCommand(' '.join(cmd), timeout = None)
      scandalOutputFiles = eval(self.filterShellOutput(output))
      for lang in scandalOutputFiles:
        outputFiles['Server '+lang] = scandalOutputFiles[lang]
      cmd = [os.path.join(self.scandalDir, 'scandal.py')]
      cmd.append('--ior=server')
      cmd.append('--serverDirs={'+server+':'+self.serverDirs[server]+'}')
      cmd.append('--includes=['+','.join(self.includes)+']')
      cmd.append('--outputSIDLFiles='+str(not self.disableOutput))
      cmd.append('--outputFiles')
      cmd.append('--logAppend')
      if 'baseDirectory' in self.argDB:
        cmd.append('--baseDirectory='+self.argDB['baseDirectory'])
      cmd.extend(source)
      (output, error, status) = self.executeShellCommand(' '.join(cmd), timeout = None)
      scandalOutputFiles = eval(self.filterShellOutput(output))
      for lang in scandalOutputFiles:
        outputFiles['Server '+lang] = scandalOutputFiles[lang]
    self.checkinServer(self.serverDirs)
    return

  def __call__(self, source, target = None):
    '''This will compile the SIDL source'''
    outputFiles           = {}
    if self.useShell:
      self.createClientShell(source, outputFiles)
      self.createServerShell(source, outputFiles)
    else:
      self.scandal.includes = self.includes
      self.scandal.targets  = source
      self.createClient(source, outputFiles)
      self.createServer(source, outputFiles)
    return ('', '', 0, outputFiles)

  def checkSetup(self):
    '''Check that this module has been specified. We assume that configure has checked its viability.'''
    import os

    if not hasattr(self, 'scandal'):
      if self.scandalDir is None:
        self.scandalDir = self.argDB['SCANDAL_DIR']
      self.scandal = self.getModule(self.scandalDir, 'scandal').Scandal(argDB = self.argDB)
      self.scandal.setup()
    return self.scandal

  def getTarget(self, source):
    '''Returns the default target for the given source file, or None'''
    return None
