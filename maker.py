import script

import os

class Make(script.Script):
  '''Template for individual project makefiles. All project makes start with a local RDict.'''
  def __init__(self, builder = None):
    import RDict
    import sys

    script.Script.__init__(self, sys.argv[1:], RDict.RDict())
    if builder is None:
      import sourceDatabase
      import config.framework

      self.framework  = config.framework.Framework(self.clArgs+['-noOutput'], self.argDB)
      self.builder    = __import__('builder').Builder(self.framework, sourceDatabase.SourceDB(self.root))
    else:
      self.builder    = builder
      self.framework  = builder.framework
    self.builder.pushLanguage('C')
    return

  def getMake(self, url):
    '''Return the Make object corresponding to the project with the given URL'''
    # FIX THIS: For now ignore RDict project info, and just use a fixed path
    import install.urlMapping

    self.logPrint('Adding project dependency: '+url)
    path   = os.path.join('/PETSc3', install.urlMapping.UrlMappingNew.getRepositoryPath(url))
    oldDir = os.getcwd()
    os.chdir(path)
    make   = self.getModule(path, 'make').Make()
    make.run()
    os.chdir(oldDir)
    return make

  def setupHelp(self, help):
    import nargs

    help = script.Script.setupHelp(self, help)
    help.addArgument('Make', 'forceConfigure', nargs.ArgBool(None, 0, 'Force a reconfiguration', isTemporary = 1))
    return help

  def setupDependencies(self, sourceDB):
    '''Override this method to setup dependencies between source files'''
    return

  def setup(self):
    script.Script.setup(self)
    self.builder.setup()
    self.setupDependencies(self.builder.sourceDB)
    return

  def shouldConfigure(self, builder, framework):
    '''Determine whether we should reconfigure
       - If the configure header or substitution files are missing
       - If -forceConfigure is given
       - If configure.py has changed
       - If the database does not contain a cached configure'''
    if framework.header and not os.path.isfile(framework.header):
      self.logPrint('Reconfiguring due to absence of configure header: '+str(framework.header))
      return 1
    if not reduce(lambda x,y: x and y, [os.path.isfile(pair[1]) for pair in framework.substFiles], True):
      self.logPrint('Reconfiguring due to absence of configure generated files: '+str([os.path.isfile(pair[1]) for pair in framework.substFiles]))
      return 1
    if self.argDB['forceConfigure']:
      self.logPrint('Reconfiguring forced')
      return 1
    if (not 'configure.py' in self.builder.sourceDB or
        not self.builder.sourceDB['configure.py'][0] == self.builder.sourceDB.getChecksum('configure.py')):
      self.logPrint('Reconfiguring due to changed configure.py')
      return 1
    if not 'configureCache' in self.argDB:
      self.logPrint('Reconfiguring due to absence of configure cache')
      return 1
    return 0

  def setupConfigure(self, framework):
    framework.header = os.path.join('include', 'config.h')
    try:
      framework.addChild(self.getModule(self.root, 'configure').Configure(framework))
    except ImportError, e:
      self.logPrint('Configure module not present: '+str(e))
      return 0
    return 1

  def configure(self, builder):
    '''Run configure if necessary and return the configuration Framework'''
    import cPickle

    if not self.setupConfigure(self.framework):
      return
    doConfigure = self.shouldConfigure(builder, self.framework)
    if not doConfigure:
      try:
        cache                  = self.argDB['configureCache']
        self.framework         = cPickle.loads(cache)
        self.framework.argDB   = self.argDB
        self.builder.framework = self.framework
        self.logPrint('Loaded configure to cache: size '+str(len(cache)))
      except cPickle.UnpicklingError, e:
        doConfigure    = 1
        self.logPrint('Invalid cached configure: '+str(e))
    if doConfigure:
      self.logPrint('Starting new configuration')
      self.framework.configure()
      self.builder.sourceDB.updateSource('configure.py')
      cache = cPickle.dumps(self.framework)
      self.argDB['configureCache'] = cache
      self.logPrint('Wrote configure to cache: size '+str(len(cache)))
    else:
      self.logPrint('Using cached configure')
      self.framework.cleanup()
    return self.framework

  def updateDependencies(self, sourceDB):
    '''Override this method to update dependencies between source files. This method saves the database'''
    sourceDB.save()
    return

  def build(self, builder):
    '''Override this method to execute all build operations. This method does nothing.'''
    return

  def run(self):
    self.setup()
    self.logPrint('Starting Build', debugSection = 'build')
    self.configure(self.builder)
    self.build(self.builder)
    self.updateDependencies(self.builder.sourceDB)
    self.logPrint('Ending Build', debugSection = 'build')
    return 1

import sets

class SIDLMake(Make):
  def __init__(self, builder = None):
    import re

    Make.__init__(self, builder)
    self.implRE       = re.compile(r'^(.*)_impl\.\w+$')
    self.dependencies = {}
    return

  def getSidl(self):
    if not hasattr(self, '_sidl'):
      self._sidl = [os.path.join(self.root, 'sidl', f) for f in filter(lambda s: os.path.splitext(s)[1] == '.sidl', os.listdir(os.path.join(self.root, 'sidl')))]
    return self._sidl
  def setSidl(self, sidl):
    self._sidl = sidl
    return
  sidl = property(getSidl, setSidl, doc = 'The list of input SIDL files')

  def getIncludes(self):
    if not hasattr(self, '_includes'):
      self._includes = []
      [self._includes.extend([os.path.join(make.getRoot(), 'sidl', f) for f in sidlFiles]) for make, sidlFiles in self.dependencies.values()]
    return self._includes
  def setIncludes(self, includes):
    self._includes = includes
    return
  includes = property(getIncludes, setIncludes, doc = 'The list of SIDL include files')

  def getClientLanguages(self):
    if not hasattr(self, '_clientLanguages'):
      self._clientLanguages = ['Python']
    return self._clientLanguages
  def setClientLanguages(self, clientLanguages):
    self._clientLanguages = clientLanguages
    return
  clientLanguages = property(getClientLanguages, setClientLanguages, doc = 'The list of client languages')

  def getServerLanguages(self):
    if not hasattr(self, '_serverLanguages'):
      self._serverLanguages = ['Python']
    return self._serverLanguages
  def setServerLanguages(self, serverLanguages):
    self._serverLanguages = serverLanguages
    return
  serverLanguages = property(getServerLanguages, setServerLanguages, doc = 'The list of server languages')

  def setupConfigure(self, framework):
    framework.require('config.libraries', None)
    framework.require('config.python', None)
    framework.require('config.ase', None)
    return Make.setupConfigure(self, framework)

  def configure(self, builder):
    framework = Make.configure(self, builder)
    self.libraries = framework.require('config.libraries', None)
    self.python    = framework.require('config.python', None)
    self.ase       = framework.require('config.ase', None)
    return framework

  def addDependency(self, url, sidlFile):
    if not url in self.dependencies:
      self.dependencies[url] = (self.getMake(url), sets.Set())
    self.dependencies[url][1].add(sidlFile)
    return

  def setupSIDL(self, builder, sidlFile):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    builder.loadConfiguration('SIDL '+baseName)
    builder.pushConfiguration('SIDL '+baseName)
    builder.pushLanguage('SIDL')
    compiler            = builder.getCompilerObject()
    compiler.clients    = self.clientLanguages
    compiler.clientDirs = dict([(lang, 'client-'+lang.lower()) for lang in self.clientLanguages])
    compiler.servers    = self.serverLanguages
    compiler.serverDirs = dict([(lang, 'server-'+lang.lower()+'-'+baseName) for lang in self.serverLanguages])
    compiler.includes   = self.includes
    builder.popLanguage()
    builder.popConfiguration()
    return

  def getSIDLClientDirectory(self, builder, sidlFile, language):
    baseName  = os.path.splitext(os.path.basename(sidlFile))[0]
    builder.pushConfiguration('SIDL '+baseName)
    builder.pushLanguage('SIDL')
    clientDir = builder.getCompilerObject().clientDirs[language]
    builder.popLanguage()
    builder.popConfiguration()
    return clientDir

  def getSIDLServerDirectory(self, builder, sidlFile, language):
    baseName  = os.path.splitext(os.path.basename(sidlFile))[0]
    builder.pushConfiguration('SIDL '+baseName)
    builder.pushLanguage('SIDL')
    serverDir = builder.getCompilerObject().serverDirs[language]
    builder.popLanguage()
    builder.popConfiguration()
    return serverDir

  def setupIOR(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    builder.loadConfiguration(language+' IOR '+baseName)
    builder.pushConfiguration(language+' IOR '+baseName)
    compiler = builder.getCompilerObject()
    compiler.includeDirectories.append(self.getSIDLServerDirectory(builder, sidlFile, language))
    for depMake, depSidlFiles in self.dependencies.values():
      for depSidlFile in depSidlFiles:
        compiler.includeDirectories.append(os.path.join(depMake.getRoot(), self.getSIDLClientDirectory(depMake.builder, depSidlFile, language)))
    builder.popConfiguration()
    return

  def setupPythonClient(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    builder.loadConfiguration(language+' Stub '+baseName)
    builder.pushConfiguration(language+' Stub '+baseName)
    compiler = builder.getCompilerObject()
    linker   = builder.getLinkerObject()
    compiler.includeDirectories.extend(self.python.include)
    compiler.includeDirectories.append(self.getSIDLClientDirectory(builder, sidlFile, language))
    for depMake, depSidlFiles in self.dependencies.values():
      for depSidlFile in depSidlFiles:
        compiler.includeDirectories.append(os.path.join(depMake.getRoot(), self.getSIDLClientDirectory(depMake.builder, depSidlFile, language)))
    linker.libraries.extend(self.ase.lib)
    linker.libraries.extend(self.python.lib)
    builder.popConfiguration()
    return

  def setupPythonSkeleton(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    builder.loadConfiguration(language+' Skeleton '+baseName)
    builder.pushConfiguration(language+' Skeleton '+baseName)
    compiler = builder.getCompilerObject()
    compiler.includeDirectories.extend(self.python.include)
    compiler.includeDirectories.append(self.getSIDLServerDirectory(builder, sidlFile, language))
    for depMake, depSidlFiles in self.dependencies.values():
      for depSidlFile in depSidlFiles:
        compiler.includeDirectories.append(os.path.join(depMake.getRoot(), self.getSIDLClientDirectory(depMake.builder, depSidlFile, language)))
    builder.popConfiguration()
    return

  def setupPythonServer(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    self.setupIOR(builder, sidlFile, language)
    self.setupPythonSkeleton(builder, sidlFile, language)
    builder.loadConfiguration(language+' Server '+baseName)
    builder.pushConfiguration(language+' Server '+baseName)
    linker   = builder.getLinkerObject()
    if not baseName == self.ase.baseName:
      linker.libraries.extend(self.ase.lib)
    linker.libraries.extend(self.python.lib)
    builder.popConfiguration()
    return

  def setupCxxClient(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    builder.loadConfiguration(language+' Stub '+baseName)
    builder.pushConfiguration(language+' Stub '+baseName)
    compiler = builder.getCompilerObject()
    linker   = builder.getLinkerObject()
    compiler.includeDirectories.append(self.getSIDLClientDirectory(builder, sidlFile, language))
    for depMake, depSidlFiles in self.dependencies.values():
      for depSidlFile in depSidlFiles:
        compiler.includeDirectories.append(os.path.join(depMake.getRoot(), self.getSIDLClientDirectory(depMake.builder, depSidlFile, language)))
    linker.libraries.extend(self.ase.lib)
    builder.popConfiguration()
    return

  def setupCxxSkeleton(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    builder.loadConfiguration(language+' Skeleton '+baseName)
    builder.pushConfiguration(language+' Skeleton '+baseName)
    compiler = builder.getCompilerObject()
    compiler.includeDirectories.append(self.getSIDLServerDirectory(builder, sidlFile, language))
    for depMake, depSidlFiles in self.dependencies.values():
      for depSidlFile in depSidlFiles:
        compiler.includeDirectories.append(os.path.join(depMake.getRoot(), self.getSIDLClientDirectory(depMake.builder, depSidlFile, language)))
    builder.popConfiguration()
    return

  def setupCxxServer(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    self.setupIOR(builder, sidlFile, language)
    self.setupCxxSkeleton(builder, sidlFile, language)
    builder.loadConfiguration(language+' Server '+baseName)
    builder.pushConfiguration(language+' Server '+baseName)
    linker   = builder.getLinkerObject()
    if not baseName == self.ase.baseName:
      linker.libraries.extend(self.ase.lib)
    builder.popConfiguration()
    return

  def buildSIDL(self, builder, sidlFile):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    config   = builder.pushConfiguration('SIDL '+baseName)
    builder.pushLanguage('SIDL')
    builder.compile([sidlFile])
    builder.popLanguage()
    builder.popConfiguration()
    builder.saveConfiguration('SIDL '+baseName)
    return config.outputFiles

  def editServer(self, builder, sidlFile):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    builder.pushConfiguration('SIDL '+baseName)
    builder.pushLanguage('SIDL')
    compiler            = builder.getCompilerObject()
    builder.popLanguage()
    builder.popConfiguration()
    for serverDir in compiler.serverDirs.values():
      for root, dirs, files in os.walk(serverDir):
        if os.path.basename(root) == 'SCCS':
          continue
        builder.versionControl.edit(builder.versionControl.getClosedFiles([os.path.join(root, f) for f in filter(lambda a: self.implRE.match(a), files)]))
    return

  def checkinServer(self, builder, sidlFile):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    builder.pushConfiguration('SIDL '+baseName)
    builder.pushLanguage('SIDL')
    compiler = builder.getCompilerObject()
    builder.popLanguage()
    builder.popConfiguration()
    vc        = builder.versionControl
    added     = 0
    reverted  = 0
    committed = 0
    for serverDir in compiler.serverDirs.values():
      for root, dirs, files in os.walk(serverDir):
        if os.path.basename(root) == 'SCCS':
          continue
        implFiles = filter(lambda a: self.implRE.match(a), files)
        added     = added or vc.add(builder.versionControl.getNewFiles([os.path.join(root, f) for f in implFiles]))
        reverted  = reverted or vc.revert(builder.versionControl.getUnchangedFiles([os.path.join(root, f) for f in implFiles]))
        committed = committed or vc.commit(builder.versionControl.getChangedFiles([os.path.join(root, f) for f in implFiles]))
    if added or committed:
      vc.changeSet()
    return

  def buildIOR(self, builder, sidlFile, language, generatedSource):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    config   = builder.pushConfiguration(language+' IOR '+baseName)
    for f in generatedSource:
      builder.compile([f])
    builder.popConfiguration()
    builder.saveConfiguration(language+' IOR '+baseName)
    if 'ELF' in config.outputFiles:
      return config.outputFiles['ELF']
    return []

  def buildPythonClient(self, builder, sidlFile, language, generatedSource):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    config   = builder.pushConfiguration(language+' Stub '+baseName)
    for f in generatedSource['Client '+language]['Cxx']:
      builder.compile([f])
      builder.link([builder.getCompilerTarget(f)], shared = 1)
    builder.popConfiguration()
    builder.saveConfiguration(language+' Stub '+baseName)
    if 'Linked ELF' in config.outputFiles:
      return config.outputFiles['Linked ELF']
    return []

  def buildPythonSkeleton(self, builder, sidlFile, language, generatedSource):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    config   = builder.pushConfiguration(language+' Skeleton '+baseName)
    for f in generatedSource:
      builder.compile([f])
    builder.popConfiguration()
    builder.saveConfiguration(language+' Skeleton '+baseName)
    if 'ELF' in config.outputFiles:
      return config.outputFiles['ELF']
    return []

  def buildPythonServer(self, builder, sidlFile, language, generatedSource):
    baseName    = os.path.splitext(os.path.basename(sidlFile))[0]
    iorObjects  = self.buildIOR(builder, sidlFile, language, generatedSource['Server IOR']['Cxx'])
    skelObjects = self.buildPythonSkeleton(builder, sidlFile, language, generatedSource['Server '+language]['Cxx'])
    config      = builder.pushConfiguration(language+' Server '+baseName)
    library     = os.path.join(os.getcwd(), 'lib', 'lib'+baseName+'.so')
    if not os.path.isdir(os.path.dirname(library)):
      os.makedirs(os.path.dirname(library))
    builder.link(iorObjects.union(skelObjects), library, shared = 1)
    builder.popConfiguration()
    builder.saveConfiguration(language+' Server '+baseName)
    if 'Linked ELF' in config.outputFiles:
      return config.outputFiles['Linked ELF']
    return []

  def build(self, builder):
    for f in self.sidl:
      self.setupSIDL(builder, f)
      for language in self.serverLanguages:
        getattr(self, 'setup'+language+'Server')(builder, f, language)
      for language in self.clientLanguages:
        getattr(self, 'setup'+language+'Client')(builder, f, language)
      self.editServer(builder, f)
      generatedSource = self.buildSIDL(builder, f)
      self.checkinServer(builder, f)
      for language in self.serverLanguages:
        getattr(self, 'build'+language+'Server')(builder, f, language, generatedSource)
      for language in self.clientLanguages:
        getattr(self, 'build'+language+'Client')(builder, f, language, generatedSource)
    return
