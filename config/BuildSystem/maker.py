import script

import os
import cPickle

class Make(script.Script):
  '''Template for individual project makefiles. All project makes start with a local RDict.'''
  def __init__(self, builder = None, clArgs = None, configureParent = None):
    import RDict
    import project
    import sys

    if clArgs is None:
      clArgs = sys.argv[1:]
    self.logName = 'build.log'
    script.Script.__init__(self, clArgs, RDict.RDict())
    if builder is None:
      import sourceDatabase
      import config.framework

      self.framework = config.framework.Framework(self.clArgs+['-noOutput'], self.argDB)
      self.framework.setConfigureParent(configureParent)
      self.framework.logName = self.logName
      self.builder   = __import__('builder').Builder(self.framework, sourceDatabase.SourceDB(self.root))
    else:
      self.builder   = builder
      self.framework = builder.framework
      self.framework.logName = self.logName
    self.configureMod = None
    self.builder.pushLanguage('C')
    return

  def getMake(self, url):
    '''Return the Make object corresponding to the project with the given URL'''
    # FIX THIS: For now ignore RDict project info, and just use a fixed path
    import install.urlMapping

    self.logPrint('Adding project dependency: '+url)
    path   = os.path.join(self.argDB['baseDirectory'], install.urlMapping.UrlMappingNew.getRepositoryPath(url))
    oldDir = os.getcwd()
    os.chdir(path)
    make   = self.getModule(path, 'make').Make()
    make.run(setupOnly = 1)
    os.chdir(oldDir)
    return make

  def setupHelp(self, help):
    import nargs

    help = script.Script.setupHelp(self, help)
    help.addArgument('Make', 'forceConfigure', nargs.ArgBool(None, 0, 'Force a reconfiguration', isTemporary = 1))
    help.addArgument('Make', 'ignoreCompileOutput', nargs.ArgBool(None, 0, 'Ignore compiler output'))
    help.addArgument('Make', 'baseDirectory', nargs.ArgDir(None, '../..', 'Directory root for all packages', isTemporary = 1))
    help.addArgument('Make', 'prefix', nargs.ArgDir(None, None, 'Root for installation of libraries and binaries', mustExist = 0, isTemporary = 1))
    return help

  def getPrefix(self):
    if not hasattr(self, '_prefix'):
      if 'prefix' in self.argDB:
        return self.argDB['prefix']
      return None
    return self._prefix
  def setPrefix(self, prefix):
    self._prefix = prefix
  prefix = property(getPrefix, setPrefix, doc = 'The installation root')

  def setupDependencies(self, sourceDB):
    '''Override this method to setup dependencies between source files'''
    return

  def updateDependencyGraph(self, graph, head):
    '''Update the directed graph with the project dependencies of head'''
    for depMake, depSidlFiles in head[0].dependencies.values():
      node = (depMake, tuple(depSidlFiles))
      for v,f in graph.vertices:
        if depMake.getRoot() == v.getRoot():
          node = (v, tuple(depSidlFiles))
          break
      graph.addEdges(head, [node])
      self.updateDependencyGraph(graph, node)
    return

  def setup(self):
    script.Script.setup(self)
    self.builder.setup()
    self.setupDependencies(self.builder.sourceDB)
    return

  def getPythonFile(self, mod, defaultFile = None):
    '''Get the Python source file associate with the module'''
    if not mod is None:
      (base, ext) = os.path.splitext(mod.__file__)
      if ext == '.pyc':
        ext = '.py'
      filename = base + ext
    elif not defaultFile is None:
      filename = defaultFile
    else:
      raise RuntimeError('Could not associate file with module '+str(mod))
    return filename

  def shouldConfigure(self, builder, framework):
    '''Determine whether we should reconfigure
       - If the configure header or substitution files are missing
       - If -forceConfigure is given
       - If configure module (usually configure.py) has changed
       - If the database does not contain a cached configure'''
    configureFile = self.getPythonFile(self.configureMod, 'configure.py')
    if framework.header and not os.path.isfile(framework.header):
      self.logPrint('Reconfiguring due to absence of configure header: '+str(framework.header))
      return 1
    if not reduce(lambda x,y: x and y, [os.path.isfile(pair[1]) for pair in framework.substFiles], 1):
      self.logPrint('Reconfiguring due to absence of configure generated files: '+str([os.path.isfile(pair[1]) for pair in framework.substFiles]))
      return 1
    if self.argDB['forceConfigure']:
      self.logPrint('Reconfiguring forced')
      return 1
    if (not configureFile in self.builder.sourceDB or
        not self.builder.sourceDB[configureFile][0] == self.builder.sourceDB.getChecksum(configureFile)):
      self.logPrint('Reconfiguring due to changed '+configureFile)
      return 1
    if not 'configureCache' in self.argDB:
      self.logPrint('Reconfiguring due to absence of configure cache')
      return 1
    return 0

  def setupConfigure(self, framework):
    framework.header = os.path.join('include', 'config.h')
    framework.cHeader = os.path.join('include', 'configC.h')
    try:
      if self.configureMod is None:
        self.configureMod = self.getModule(self.root, 'configure')
      self.configureObj = self.configureMod.Configure(framework)
      self.logPrint('Configure module found in '+self.configureObj.root)
      self.configureObj.argDB = self.argDB
      self.configureObj.setup()
      self.configureObj.setupPackageDependencies(framework)
      self.configureObj.setupDependencies(framework)
      framework.addChild(self.configureObj)
    except ImportError, e:
      self.configureObj = None
      self.logPrint('Configure module not present: '+str(e))
      return 0
    return 1

  def configure(self, builder):
    '''Run configure if necessary and return the configuration Framework'''
    if not self.setupConfigure(self.framework):
      return
    doConfigure = self.shouldConfigure(builder, self.framework)
    if not doConfigure:
      framework = self.loadConfigure()
      if framework is None:
        doConfigure = 1
      else:
        self.framework         = framework
        self.builder.framework = self.framework
        if not self.configureObj is None:
          self.configureObj = self.framework.require(self.configureMod.__name__, None)
    if doConfigure:
      self.logPrint('Starting new configuration')
      self.framework.configure()
      self.builder.sourceDB.updateSource(self.getPythonFile(self.configureMod))
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

  def setupBuild(self, builder):
    '''Override this method to execute all build setup operations. This method does nothing.'''
    return

  def build(self, builder, setupOnly = 0):
    '''Override this method to execute all build operations. This method does nothing.'''
    return

  def install(self, builder, argDB):
    '''Override this method to execute all install operations. This method does nothing.'''
    return

  def outputBanner(self):
    import time

    self.log.write(('='*80)+'\n')
    self.log.write(('='*80)+'\n')
    self.log.write('Starting Build Run at '+time.ctime(time.time())+'\n')
    self.log.write('Build Options: '+str(self.clArgs)+'\n')
    self.log.write('Working directory: '+os.getcwd()+'\n')
    self.log.write(('='*80)+'\n')
    return

  def executeSection(self, section, *args):
    import time

    self.log.write(('='*80)+'\n')
    self.logPrint('SECTION: '+str(section.im_func.func_name)+' in '+self.getRoot()+' from '+str(section.im_class.__module__)+'('+str(section.im_func.func_code.co_filename)+':'+str(section.im_func.func_code.co_firstlineno)+') at '+time.ctime(time.time()), debugSection = 'screen', indent = 0)
    if section.__doc__: self.logWrite('  '+section.__doc__+'\n')
    return section(*args)

  def run(self, setupOnly = 0):
    self.setup()
    try:
      self.logPrint('Starting Build', debugSection = 'build')
      self.executeSection(self.configure, self.builder)
      self.executeSection(self.build, self.builder, setupOnly)
      self.executeSection(self.updateDependencies, self.builder.sourceDB)
      self.executeSection(self.install, self.builder, self.argDB)
      self.logPrint('Ending Build', debugSection = 'build')
    except Exception, e:
      import sys, traceback
      self.logPrint('************************************ ERROR **************************************')
      traceback.print_tb(sys.exc_info()[2], file = self.log)
      raise
    return 1

try:
  import sets
except ImportError:
  import config.setsBackport as sets

class struct:
  '''Container class'''

class BasicMake(Make):
  '''A basic make template that acts much like a traditional makefile'''
  languageNames = {'C': 'C', 'Cxx': 'Cxx', 'FC': 'Fortran', 'Python': 'Python'}

  def __init__(self, implicitRoot = 0, configureParent = None, module = None):
    '''Setup the library and driver source descriptions'''
    if not implicitRoot:
      self.root = os.getcwd()
    Make.__init__(self, configureParent = configureParent)
    self.lib = {}
    self.dylib = {}
    self.bin = {}
    if not module is None:
      self.module = module
    return

  def setupHelp(self, help):
    import nargs

    help = Make.setupHelp(self, help)
    help.addArgument('basicMake', 'libdir', nargs.ArgDir(None, 'lib', 'Root for installation of libraries', mustExist = 0, isTemporary = 1))
    help.addArgument('basicMake', 'bindir', nargs.ArgDir(None, 'bin', 'Root for installation of executables', mustExist = 0, isTemporary = 1))
    return help

  def getMakeModule(self):
    if not hasattr(self, '_module'):
      import sys
      d = sys.modules['__main__']
      if os.path.basename(d.__file__) == 'pdb.py':
        sys.path.insert(0, '.')
        import make
        d = sys.modules['make']
      elif not 'Make' in dir(d):
        d = sys.modules['make']
      return d
    return self._module
  def setMakeModule(self, module):
    self._module = module
    return
  module = property(getMakeModule, setMakeModule, doc = 'The make module for this build')

  def classifySource(self, srcList):
    src = {}
    for f in srcList:
      base, ext = os.path.splitext(f)
      if ext in ['.c', '.h']:
        if not 'C' in src:
          src['C'] = []
        src['C'].append(f)
      elif ext in ['.cc', '.hh', '.C', '.cpp', '.cxx']:
        if not 'Cxx' in src:
          src['Cxx'] = []
        src['Cxx'].append(f)
      elif ext in ['.F', '.f']:
        if not 'FC' in src:
          src['FC'] = []
        src['FC'].append(f)
      elif ext in ['.py']:
        if not 'Python' in src:
          src['Python'] = []
        src['Python'].append(f)
    return src

  def classifyIncludes(self, incList):
    inc = {}
    for f in incList:
      base, ext = os.path.splitext(f)
      if ext in ['.h']:
        if not 'C' in inc:
          inc['C'] = []
        inc['C'].append(f)
      elif ext in ['.hh']:
        if not 'Cxx' in inc:
          inc['Cxx'] = []
        inc['Cxx'].append(f)
    return inc

  def parseDocString(self, docstring, defaultName = None):
    parts = docstring.split(':', 1)
    if len(parts) < 2:
      if defaultName is None:
        raise RuntimeError('No target specified in docstring')
      name = defaultName
      src = parts[0].split()
    else:
      name = parts[0]
      src = parts[1].split()
    return (name, self.classifySource(src), self.classifyIncludes(src))

  def setupConfigure(self, framework):
    '''We always want to configure'''
    Make.setupConfigure(self, framework)
    framework.require('config.setCompilers', self.configureObj)
    framework.require('config.compilers', self.configureObj)
    framework.require('config.libraries', self.configureObj)
    framework.require('config.headers', self.configureObj)
    return 1

  def configure(self, builder):
    framework = Make.configure(self, builder)
    self.setCompilers = framework.require('config.setCompilers', None)
    self.compilers = framework.require('config.compilers', None)
    self.libraries = framework.require('config.libraries', None)
    self.headers = framework.require('config.headers', None)
    return framework

  def getImplicitLibraries(self):
    d = self.module
    for name in dir(d):
      if not name.startswith('lib_'):
        continue
      func = getattr(d, name)
      lib = struct()
      lib.name, lib.src, lib.inc = self.parseDocString(func.__doc__, name[4:])
      params = func(self)
      lib.includes, lib.libs = params[0:2]
      if (len(params) == 3): lib.flags = params[2]
      lib.configuration = name[4:]
      self.logPrint('Found configuration '+lib.configuration+' for library '+lib.name)
      self.logPrint('  includes '+str(lib.includes)+' libraries '+str(lib.libs))
      self.lib[lib.name] = lib
    return

  def getImplicitDynamicLibraries(self):
    d = self.module
    for name in dir(d):
      if not name.startswith('dylib_'):
        continue
      func = getattr(d, name)
      lib = struct()
      lib.name, lib.src, lib.inc = self.parseDocString(func.__doc__, name[6:])
      params = func(self)
      lib.includes, lib.libs = params[0:2]
      if (len(params) == 3): lib.flags = params[2]
      lib.configuration = name[6:]
      self.logPrint('Found configuration '+lib.configuration+' for dynamic library '+lib.name)
      self.logPrint('  includes '+str(lib.includes)+' libraries '+str(lib.libs))
      self.dylib[lib.name] = lib
    return

  def getImplicitExecutables(self):
    d = self.module
    for name in dir(d):
      if not name.startswith('bin_'):
        continue
      func = getattr(d, name)
      bin = struct()
      bin.name, bin.src, bin.inc = self.parseDocString(func.__doc__, name[4:])
      params = func(self)
      bin.includes, bin.libs = params[0:2]
      if (len(params) == 3): bin.flags = params[2]
      bin.configuration = name[4:]
      self.bin[bin.name] = bin
    return

  def setupDirectories(self, builder):
    if self.prefix is None:
      self.logPrint('ERROR: prefix is None')
      self.libDir = os.path.abspath(self.argDB['libdir'])
      self.binDir = os.path.abspath(self.argDB['bindir'])
    else:
      self.logPrint('prefix '+self.prefix+' libDir '+self.argDB['libdir']+' totdir '+os.path.join(self.prefix, self.argDB['libdir']))
      self.libDir = os.path.abspath(os.path.join(self.prefix, self.argDB['libdir']))
      self.binDir = os.path.abspath(os.path.join(self.prefix, self.argDB['bindir']))
    self.logPrint('Library directory is '+self.libDir)
    self.logPrint('Executable directory is '+self.binDir)
    return

  def setupLibraryDirectories(self, builder):
    '''Determine the directories for source includes, libraries, and binaries'''
    languages = sets.Set()
    [languages.update(lib.src.keys()) for lib in self.lib.values()+self.dylib.values()]
    self.srcDir = {}
    self.includeDir = {}
    for language in languages:
      self.srcDir[language] = os.path.abspath(os.path.join('src', self.languageNames[language].lower()))
      self.logPrint('Source directory for '+language+' is '+self.srcDir[language])
      if self.prefix is None:
        self.includeDir[language] = os.path.abspath('include')
      else:
        self.includeDir[language] = os.path.abspath(os.path.join(self.prefix, 'include'))
      self.logPrint('Include directory for '+language+' is '+self.includeDir[language])
    return

  def setupLibraries(self, builder):
    '''Configures the builder for libraries'''
    for lib in self.lib.values():
      self.logPrint('Configuring library '+lib.name)
      languages = sets.Set(lib.src.keys())
      builder.pushConfiguration(lib.configuration)
      for language in languages:
        builder.pushLanguage(language)
        if hasattr(lib, 'flags'):
          builder.setCompilerFlags(' '.join(lib.flags))
        compiler = builder.getCompilerObject()
        lib.includes = filter(lambda inc: inc, lib.includes)
        self.logPrint('  Adding includes '+str(lib.includes))
        compiler.includeDirectories.update(lib.includes)
        builder.popLanguage()
      linker = builder.getSharedLinkerObject()
      for l in lib.libs:
        if not l: continue
        if isinstance(l, str):
          self.logPrint('  Adding library '+str(l))
          linker.libraries.add(l)
        else:
          self.logPrint('  Adding library '+os.path.join(self.libDir, l.name+'.'+self.setCompilers.sharedLibraryExt))
          linker.libraries.add(os.path.join(self.libDir, l.name+'.'+self.setCompilers.sharedLibraryExt))
      linker.libraries.update(self.compilers.flibs)
      if self.libraries.math:
        linker.libraries.update(self.libraries.math)
      if self.setCompilers.explicitLibc:
        linker.libraries.update(self.setCompilers.explicitLibc)
      builder.popConfiguration()
    return

  def setupDynamicLibraries(self, builder):
    '''Configures the builder for dynamic libraries'''
    for lib in self.dylib.values():
      self.logPrint('Configuring dynamic library '+lib.name)
      languages = sets.Set(lib.src.keys())
      builder.pushConfiguration(lib.configuration)
      for language in languages:
        builder.pushLanguage(language)
        if hasattr(lib, 'flags'):
          builder.setCompilerFlags(' '.join(lib.flags))
        compiler = builder.getCompilerObject()
        lib.includes = filter(lambda inc: inc, lib.includes)
        self.logPrint('  Adding includes '+str(lib.includes))
        compiler.includeDirectories.update(lib.includes)
        builder.popLanguage()
      linker = builder.getDynamicLinkerObject()
      for l in lib.libs:
        if not l: continue
        if isinstance(l, str):
          self.logPrint('  Adding library '+str(l))
          linker.libraries.add(l)
        else:
          self.logPrint('  Adding library '+os.path.join(self.libDir, l.name+'.'+self.setCompilers.dynamicLibraryExt))
          linker.libraries.add(os.path.join(self.libDir, l.name+'.'+self.setCompilers.dynamicLibraryExt))
      linker.libraries.update(self.compilers.flibs)
      if self.libraries.math:
        linker.libraries.update(self.libraries.math)
      if self.setCompilers.explicitLibc:
        linker.libraries.update(self.setCompilers.explicitLibc)
      builder.popConfiguration()
    return

  def setupExecutables(self, builder):
    '''Configures the builder for the executable'''
    for bin in self.bin.values():
      self.logPrint('Configuring executable '+bin.name)
      languages = sets.Set(bin.src.keys())
      builder.pushConfiguration(bin.configuration)
      for language in languages:
        builder.pushLanguage(language)
        if hasattr(bin, 'flags'):
          builder.setCompilerFlags(' '.join(bin.flags))
        compiler = builder.getCompilerObject()
        bin.includes = filter(lambda inc: inc, bin.includes)
        self.logPrint('  Adding includes '+str(bin.includes))
        compiler.includeDirectories.update(bin.includes)
        builder.popLanguage()
      linker = builder.getLinkerObject()
      for l in bin.libs:
        if not l: continue
        if isinstance(l, str):
          self.logPrint('  Adding library '+str(l))
          linker.libraries.add(l)
        else:
          self.logPrint('  Adding library '+os.path.join(self.libDir, l.name+'.'+self.setCompilers.sharedLibraryExt))
          linker.libraries.add(os.path.join(self.libDir, l.name+'.'+self.setCompilers.sharedLibraryExt))
      linker.libraries.update(self.compilers.fmainlibs)
      linker.libraries.update(self.compilers.flibs)
      if self.libraries.math:
        linker.libraries.update(self.libraries.math)
      builder.popConfiguration()
    return

  def buildDirectories(self, builder):
    '''Create the necessary directories'''
    languages = sets.Set()
    [languages.update(lib.src.keys()) for lib in self.lib.values()+self.dylib.values()]
    for language in languages:
      if not os.path.isdir(self.includeDir[language]):
        os.mkdir(self.includeDir[language])
        self.logPrint('Created include directory '+self.includeDir[language])
    if not os.path.isdir(self.libDir):
      os.mkdir(self.libDir)
      self.logPrint('Created library directory '+self.libDir)
    if not os.path.isdir(self.binDir):
      os.mkdir(self.binDir)
      self.logPrint('Created executable directory '+self.binDir)
    return

  def buildLibraries(self, builder):
    '''Builds the libraries'''
    for lib in self.lib.values():
      self.logPrint('Building library: '+lib.name)
      builder.pushConfiguration(lib.configuration)
      objects = []
      for language in lib.src:
        builder.pushLanguage(language)
        sources = [os.path.join(self.srcDir, self.srcDir[language], f) for f in lib.src[language]]
        for f in sources:
          builder.compile([f])
        objects.extend([self.builder.getCompilerTarget(f) for f in sources if not self.builder.getCompilerTarget(f) is None])
        builder.popLanguage()
      builder.link(objects, os.path.join(self.libDir, lib.name+'.'+self.setCompilers.sharedLibraryExt), shared = 1)
      builder.popConfiguration()
    return

  def buildDynamicLibraries(self, builder):
    '''Builds the dynamic libraries'''
    for lib in self.dylib.values():
      self.logPrint('Building dynamic library: '+lib.name)
      builder.pushConfiguration(lib.configuration)
      objects = []
      for language in lib.src:
        builder.pushLanguage(language)
        sources = [os.path.join(self.srcDir, self.srcDir[language], f) for f in lib.src[language]]
        for f in sources:
          builder.compile([f])
        objects.extend([self.builder.getCompilerTarget(f) for f in sources if not self.builder.getCompilerTarget(f) is None])
        builder.popLanguage()
      builder.link(objects, os.path.join(self.libDir, lib.name+'.'+self.setCompilers.dynamicLibraryExt), shared = 'dynamic')
      builder.popConfiguration()
    return

  def buildExecutables(self, builder):
    '''Builds the executables'''
    for bin in self.bin.values():
      source = []
      builder.pushConfiguration(bin.configuration)
      for language in bin.src:
        builder.pushLanguage(language)
        source.extend([os.path.join(self.srcDir, self.srcDir[language], f) for f in bin.src[language]])
        for f in source:
          builder.compile([f])
        builder.popLanguage()
      # Note that we popLanguage before linking, since the linker is configure independently of the exe source language
      # If the executable is in Fortran, we need to add the appropriate runtime libs
      builder.link([builder.getCompilerTarget(f) for f in source], os.path.join(self.binDir, bin.name))
      builder.popConfiguration()
    return

  def setupBuild(self, builder):
    self.executeSection(self.setupDirectories, builder)
    self.getImplicitLibraries()
    self.getImplicitDynamicLibraries()
    self.getImplicitExecutables()
    self.executeSection(self.setupLibraryDirectories, builder)
    self.executeSection(self.setupLibraries, builder)
    self.executeSection(self.setupDynamicLibraries, builder)
    self.executeSection(self.setupExecutables, builder)
    return

  def build(self, builder, setupOnly):
    self.setupBuild(builder)
    if setupOnly:
      return
    self.executeSection(self.buildDirectories, builder)
    self.executeSection(self.buildLibraries, builder)
    self.executeSection(self.buildDynamicLibraries, builder)
    self.executeSection(self.buildExecutables, builder)
    return

  def installIncludes(self, builder):
    import shutil
    for lib in self.lib.values()+self.dylib.values():
      self.logPrint('Installing library: '+lib.name)
      for language in lib.inc:
        for inc in lib.inc[language]:
          installInc = os.path.join(self.includeDir[language], os.path.basename(inc))
          if os.path.isfile(installInc):
            os.remove(installInc)
          self.logPrint('Installing '+inc+' into '+installInc)
          shutil.copy(os.path.join(self.srcDir[language], inc), installInc)
    return

  def install(self, builder, argDB):
    self.executeSection(self.installIncludes, builder)
    return

class SIDLMake(Make):
  def __init__(self, builder = None):
    import re

    Make.__init__(self, builder)
    self.implRE       = re.compile(r'^((.*)_impl\.(c|h|py)|__init__\.py)$')
    self.dependencies = {}
    return

  def getSidl(self):
    if not hasattr(self, '_sidl'):
      import graph
      g = graph.DirectedGraph([os.path.join(self.root, 'sidl', f) for f in filter(lambda s: os.path.splitext(s)[1] == '.sidl', os.listdir(os.path.join(self.root, 'sidl')))])
      for vertex in g.vertices:
        g.addEdges(vertex, [os.path.join(self.root, dep) for dep in self.builder.sourceDB.getDependencies(vertex)])
      self._sidl = [v for v in g.topologicalSort(g)]
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

  def setupHelp(self, help):
    import nargs

    help = Make.setupHelp(self, help)
    help.addArgument('SIDLMake', 'bootstrap', nargs.ArgBool(None, 0, 'Generate the boostrap client', isTemporary = 1))
    help.addArgument('SIDLMake', 'outputSIDLFiles', nargs.ArgBool(None, 1, 'Write generated files to disk', isTemporary = 1))
    help.addArgument('SIDLMake', 'excludeLanguages=<languages>', nargs.Arg(None, [], 'Do not load configurations from RDict for the given languages', isTemporary = 1))
    help.addArgument('SIDLMake', 'excludeBasenames=<names>', nargs.Arg(None, [], 'Do not load configurations from RDict for these SIDL base names', isTemporary = 1))
    return help

  def setup(self):
    Make.setup(self)
    import graph
    self.dependencyGraph = graph.DirectedGraph()
    self.updateDependencyGraph(self.dependencyGraph, (self, tuple(self.sidl)))
    return

  def setupConfigure(self, framework):
    framework.require('config.libraries', None)
    framework.require('config.python', None)
    framework.require('config.ase', None)
    return Make.setupConfigure(self, framework)

  def configure(self, builder):
    import graph
    framework = Make.configure(self, builder)
    if framework is None:
      for depMake, depSidlFiles in graph.DirectedGraph.topologicalSort(self.dependencyGraph, outEdges = 0):
        if depMake is self: continue
        self.logPrint('Loading configure for '+depMake.getRoot())
        framework = depMake.loadConfigure()
        if not framework is None:
          self.framework         = framework
          self.builder.framework = framework
          break
    if framework is None:
      raise RuntimeError('Could not find a configure framework')
    self.compilers = framework.require('config.compilers', None)
    self.libraries = framework.require('config.libraries', None)
    self.python    = framework.require('config.python', None)
    self.ase       = framework.require('config.ase', None)
    if self.configureObj is None:
      if self.configureMod is None:
        self.configureObj = framework.require('configure', None)
      else:
        self.configureObj = framework.require(self.configureMod.__name__, None)
    return framework

  def addDependency(self, url, sidlFile, make = None):
    if not url in self.dependencies:
      if make is None:
        make = self.getMake(url)
      self.dependencies[url] = (make, sets.Set())
      for depMake, depSidlFiles in make.dependencies.values():
        [self.addDependency(depMake.project.getUrl(), depSidlFile, depMake) for depSidlFile in depSidlFiles]
    self.dependencies[url][1].add(sidlFile)
    return

  def loadConfiguration(self, builder, name):
    if len(self.argDB['excludeLanguages']) and len(self.argDB['excludeBasenames']):
      for language in self.argDB['excludeLanguages']:
        if name.startswith(language):
          for basename in self.argDB['excludeBasenames']:
            if name.endswith(basename):
              return
    elif len(self.argDB['excludeLanguages']):
      for language in self.argDB['excludeLanguages']:
        if name.startswith(language):
          return
    elif len(self.argDB['excludeBasenames']):
      for basename in self.argDB['excludeBasenames']:
        if name.endswith(basename):
          return
    builder.loadConfiguration(name)
    return

  def getSIDLClientDirectory(self, builder, sidlFile, language):
    baseName  = os.path.splitext(os.path.basename(sidlFile))[0]
    clientDir = None
    builder.pushConfiguration('SIDL '+baseName)
    builder.pushLanguage('SIDL')
    if language in builder.getCompilerObject().clientDirs:
      clientDir = builder.getCompilerObject().clientDirs[language]
    builder.popLanguage()
    builder.popConfiguration()
    return clientDir

  def getSIDLServerDirectory(self, builder, sidlFile, language):
    baseName  = os.path.splitext(os.path.basename(sidlFile))[0]
    serverDir = None
    builder.pushConfiguration('SIDL '+baseName)
    builder.pushLanguage('SIDL')
    if language in builder.getCompilerObject().serverDirs:
      serverDir = builder.getCompilerObject().serverDirs[language]
    builder.popLanguage()
    builder.popConfiguration()
    return serverDir

  def addDependencyIncludes(self, compiler, language):
    for depMake, depSidlFiles in self.dependencies.values():
      for depSidlFile in depSidlFiles:
        try:
          if hasattr(compiler.includeDirectories, 'add'):
            compiler.includeDirectories.add(os.path.join(depMake.getRoot(), self.getSIDLClientDirectory(depMake.builder, depSidlFile, language)))
          else:
            if not self.getSIDLClientDirectory(depMake.builder, depSidlFile, language):
              raise RuntimeError('Cannot determine '+language+' client directory for '+str(depMake.root)+'('+depSidlFile+')')
            compiler.includeDirectories[language].add(os.path.join(depMake.getRoot(), self.getSIDLClientDirectory(depMake.builder, depSidlFile, language)))
        except KeyError, e:
          if e.args[0] == language:
            self.logPrint('Dependency '+depSidlFile+' has no client for '+language, debugSection = 'screen')
          else:
            raise e
    return

  def addDependencyLibraries(self, linker, language):
    for depMake, depSidlFiles in self.dependencies.values():
      for depSidlFile in depSidlFiles:
        self.logPrint('Checking dependency '+depSidlFile+' for a '+language+' client', debugSection = 'build')
        try:
          clientConfig = depMake.builder.pushConfiguration(language+' Stub '+os.path.splitext(os.path.basename(depSidlFile))[0])
          if 'Linked ELF' in clientConfig.outputFiles:
            files = sets.Set([os.path.join(depMake.getRoot(), lib) for lib in clientConfig.outputFiles['Linked ELF']])
            self.logPrint('Adding '+str(files)+'from dependency '+depSidlFile, debugSection = 'build')
            linker.libraries.update(files)
          depMake.builder.popConfiguration()
        except KeyError, e:
          if e.args[0] == language:
            self.logPrint('Dependency '+depSidlFile+' has no client for '+language, debugSection = 'screen')
          else:
            raise e
    return

  def setupSIDL(self, builder, sidlFile):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    self.loadConfiguration(builder, 'SIDL '+baseName)
    builder.pushConfiguration('SIDL '+baseName)
    builder.pushLanguage('SIDL')
    compiler            = builder.getLanguageProcessor().getCompilerObject(builder.language[-1])
    compiler.scandalDir = self.ase.scandalDir
    compiler.checkSetup()
    compiler.clients    = self.clientLanguages
    compiler.clientDirs = dict([(lang, 'client-'+lang.lower()) for lang in self.clientLanguages])
    compiler.servers    = self.serverLanguages
    compiler.serverDirs = dict([(lang, 'server-'+lang.lower()+'-'+baseName) for lang in self.serverLanguages])
    compiler.includes   = self.includes+list(builder.sourceDB.getDependencies(sidlFile))
    for language in self.serverLanguages:
      if not hasattr(compiler, 'includeDirectories'):
        compiler.includeDirectories = {}
      if not language in compiler.includeDirectories:
        compiler.includeDirectories[language] = sets.Set()
      self.addDependencyIncludes(compiler, language)
      compiler.includeDirectories[language].add(os.path.join(self.getRoot(), self.getSIDLClientDirectory(builder, sidlFile, language)))
      compiler.includeDirectories[language].add(os.path.join(self.getRoot(), self.getSIDLServerDirectory(builder, sidlFile, language)))
    compiler.disableOutput = not self.argDB['outputSIDLFiles']
    builder.popLanguage()
    builder.popConfiguration()
    return

  def setupIOR(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    self.loadConfiguration(builder, language+' IOR '+baseName)
    builder.pushConfiguration(language+' IOR '+baseName)
    compiler = builder.getCompilerObject()
    compiler.includeDirectories.add(self.getSIDLServerDirectory(builder, sidlFile, language))
    for depFile in builder.sourceDB.getDependencies(sidlFile):
      dir = self.getSIDLServerDirectory(builder, depFile, language)
      if not dir is None:
        compiler.includeDirectories.add(dir)
    self.addDependencyIncludes(compiler, language)
    builder.popConfiguration()
    return

  def setupPythonClient(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    self.loadConfiguration(builder, language+' Stub '+baseName)
    builder.pushConfiguration(language+' Stub '+baseName)
    compiler = builder.getCompilerObject()
    linker   = builder.getSharedLinkerObject()
    compiler.includeDirectories.update(self.python.include)
    compiler.includeDirectories.add(self.getSIDLClientDirectory(builder, sidlFile, language))
    self.addDependencyIncludes(compiler, language)
    linker.libraries.clear()
    linker.libraries.update(self.ase.lib)
    linker.libraries.update(self.python.lib)
    builder.popConfiguration()
    return

  def setupPythonSkeleton(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    self.loadConfiguration(builder, language+' Skeleton '+baseName)
    builder.pushConfiguration(language+' Skeleton '+baseName)
    compiler = builder.getCompilerObject()
    compiler.includeDirectories.update(self.python.include)
    compiler.includeDirectories.add(self.getSIDLServerDirectory(builder, sidlFile, language))
    for depFile in builder.sourceDB.getDependencies(sidlFile):
      dir = self.getSIDLServerDirectory(builder, depFile, language)
      if not dir is None:
        compiler.includeDirectories.add(dir)
    self.addDependencyIncludes(compiler, language)
    builder.popConfiguration()
    return

  def setupPythonServer(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    self.setupIOR(builder, sidlFile, language)
    self.setupPythonSkeleton(builder, sidlFile, language)
    self.loadConfiguration(builder, language+' Server '+baseName)
    builder.pushConfiguration(language+' Server '+baseName)
    linker   = builder.getSharedLinkerObject()
    linker.libraries.clear()
    if not baseName == self.ase.baseName:
      linker.libraries.update(self.ase.lib)
    linker.libraries.update(self.python.lib)
    builder.popConfiguration()
    return

  def setupCxxClient(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    self.loadConfiguration(builder, language+' Stub '+baseName)
    builder.pushConfiguration(language+' Stub '+baseName)
    compiler = builder.getCompilerObject()
    linker   = builder.getSharedLinkerObject()
    compiler.includeDirectories.add(self.getSIDLClientDirectory(builder, sidlFile, language))
    self.addDependencyIncludes(compiler, language)
    linker.libraries.clear()
    linker.libraries.update(self.ase.lib)
    builder.popConfiguration()
    return

  def setupCxxSkeleton(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    self.loadConfiguration(builder, language+' Skeleton '+baseName)
    builder.pushConfiguration(language+' Skeleton '+baseName)
    compiler = builder.getCompilerObject()
    compiler.includeDirectories.add(self.getSIDLServerDirectory(builder, sidlFile, language))
    compiler.includeDirectories.add(self.getSIDLClientDirectory(builder, sidlFile, language))
    for depFile in builder.sourceDB.getDependencies(sidlFile):
      dir = self.getSIDLServerDirectory(builder, depFile, language)
      if not dir is None:
        compiler.includeDirectories.add(dir)
    self.addDependencyIncludes(compiler, language)
    builder.popConfiguration()
    return

  def setupCxxServer(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    self.setupIOR(builder, sidlFile, language)
    self.setupCxxSkeleton(builder, sidlFile, language)
    self.loadConfiguration(builder, language+' Server '+baseName)
    builder.pushConfiguration(language+' Server '+baseName)
    linker   = builder.getSharedLinkerObject()
    linker.libraries.clear()
    self.addDependencyLibraries(linker, language)
    if not baseName == self.ase.baseName:
      linker.libraries.update(self.ase.lib)
    builder.popConfiguration()
    return

  def setupBootstrapClient(self, builder, sidlFile, language):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    self.loadConfiguration(builder, language+' Stub '+baseName)
    builder.pushConfiguration(language+' Stub '+baseName)
    builder.popConfiguration()
    return

  def buildSIDL(self, builder, sidlFile):
    self.logPrint('Building '+sidlFile)
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    config   = builder.pushConfiguration('SIDL '+baseName)
    builder.pushLanguage('SIDL')
    builder.compile([sidlFile])
    builder.popLanguage()
    builder.popConfiguration()
    builder.saveConfiguration('SIDL '+baseName)
    self.logPrint('generatedFiles: '+str(config.outputFiles), debugSection = 'sidl')
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
    return sets.Set()

  def buildPythonClient(self, builder, sidlFile, language, generatedSource):
    if not 'Client '+language in generatedSource:
      return sets.Set()
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    config   = builder.pushConfiguration(language+' Stub '+baseName)
    for f in generatedSource['Client '+language]['Cxx']:
      builder.compile([f])
      builder.link([builder.getCompilerTarget(f)], builder.getSharedLinkerTarget(builder.getCompilerTarget(f), 1, None), shared = 1)
    builder.popConfiguration()
    builder.saveConfiguration(language+' Stub '+baseName)
    if 'Linked ELF' in config.outputFiles:
      return config.outputFiles['Linked ELF']
    return sets.Set()

  def buildPythonSkeleton(self, builder, sidlFile, language, generatedSource):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    config   = builder.pushConfiguration(language+' Skeleton '+baseName)
    for f in generatedSource:
      builder.compile([f])
    builder.popConfiguration()
    builder.saveConfiguration(language+' Skeleton '+baseName)
    if 'ELF' in config.outputFiles:
      return config.outputFiles['ELF']
    return sets.Set()

  def buildPythonServer(self, builder, sidlFile, language, generatedSource):
    if not 'Server IOR Python' in generatedSource:
      return sets.Set()
    baseName    = os.path.splitext(os.path.basename(sidlFile))[0]
    iorObjects  = self.buildIOR(builder, sidlFile, language, generatedSource['Server IOR Python']['Cxx'])
    skelObjects = self.buildPythonSkeleton(builder, sidlFile, language, generatedSource['Server '+language]['Cxx'])
    config      = builder.pushConfiguration(language+' Server '+baseName)
    library     = os.path.join(os.getcwd(), 'lib', 'lib-'+language.lower()+'-'+baseName+'.so')
    if not os.path.isdir(os.path.dirname(library)):
      os.makedirs(os.path.dirname(library))
    builder.link(iorObjects.union(skelObjects), library, shared = 1)
    builder.popConfiguration()
    builder.saveConfiguration(language+' Server '+baseName)
    if 'Linked ELF' in config.outputFiles:
      return config.outputFiles['Linked ELF']
    return sets.Set()

  def buildCxxClient(self, builder, sidlFile, language, generatedSource):
    if not 'Client '+language in generatedSource:
      return sets.Set()
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    config   = builder.pushConfiguration(language+' Stub '+baseName)
    for f in generatedSource['Client '+language]['Cxx']:
      builder.compile([f])
      if builder.getCompilerTarget(f):
        builder.link([builder.getCompilerTarget(f)], builder.getSharedLinkerTarget(builder.getCompilerTarget(f), 1, None), shared = 1)
    builder.popConfiguration()
    builder.saveConfiguration(language+' Stub '+baseName)
    if 'Linked ELF' in config.outputFiles:
      return config.outputFiles['Linked ELF']
    return sets.Set()

  def buildCxxImplementation(self, builder, sidlFile, language, generatedSource):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    config   = builder.pushConfiguration(language+' Skeleton '+baseName)
    for f in generatedSource:
      builder.compile([f])
    builder.popConfiguration()
    builder.saveConfiguration(language+' Skeleton '+baseName)
    if 'ELF' in config.outputFiles:
      return config.outputFiles['ELF']
    return sets.Set()

  def buildCxxServer(self, builder, sidlFile, language, generatedSource):
    baseName    = os.path.splitext(os.path.basename(sidlFile))[0]
    iorObjects  = self.buildIOR(builder, sidlFile, language, generatedSource['Server IOR Cxx']['Cxx'])
    implObjects = self.buildCxxImplementation(builder, sidlFile, language, generatedSource['Server '+language]['Cxx'])
    config      = builder.pushConfiguration(language+' Server '+baseName)
    library     = os.path.join(os.getcwd(), 'lib', 'lib-'+language.lower()+'-'+baseName+'.so')
    linker      = builder.getSharedLinkerObject()
    if not os.path.isdir(os.path.dirname(library)):
      os.makedirs(os.path.dirname(library))
    for depSidlFile in builder.sourceDB.getDependencies(sidlFile)+(sidlFile,):
      self.logPrint('Checking dependency '+depSidlFile+' for a '+language+' client', debugSection = 'build')
      clientConfig = builder.pushConfiguration(language+' Stub '+os.path.splitext(os.path.basename(depSidlFile))[0])
      if 'Linked ELF' in clientConfig.outputFiles:
        files = [os.path.join(self.getRoot(), f) for f in clientConfig.outputFiles['Linked ELF']]
        self.logPrint('Adding '+str(files)+' from dependency '+depSidlFile, debugSection = 'build')
        linker.libraries.update(files)
      builder.popConfiguration()
    builder.link(iorObjects.union(implObjects), library, shared = 1)
    builder.popConfiguration()
    builder.saveConfiguration(language+' Server '+baseName)
    if 'Linked ELF' in config.outputFiles:
      return config.outputFiles['Linked ELF']
    return sets.Set()

  def buildBootstrapClient(self, builder, sidlFile, language, generatedSource):
    baseName = os.path.splitext(os.path.basename(sidlFile))[0]
    config   = builder.pushConfiguration(language+' Stub '+baseName)
    builder.popConfiguration()
    builder.saveConfiguration(language+' Stub '+baseName)
    return sets.Set()

  def setupBootstrap(self, builder):
    '''If bootstrap flag is enabled, setup varaibles to generate the bootstrap client'''
    if self.argDB['bootstrap']:
      self.serverLanguages = []
      self.clientLanguages = ['Bootstrap']
      builder.shouldCompile.force(self.sidl)
    return

  def setupBuild(self, builder, f):
    self.executeSection(self.setupSIDL, builder, f)
    for language in self.serverLanguages:
      self.executeSection(getattr(self, 'setup'+language+'Server'), builder, f, language)
    for language in self.clientLanguages:
      self.executeSection(getattr(self, 'setup'+language+'Client'), builder, f, language)
    return

  def build(self, builder, setupOnly = 0):
    import shutil

    self.setupBootstrap(builder)
    for f in self.sidl:
      self.setupBuild(builder, f)
      if not setupOnly:
        # We here require certain keys to be present in generatedSource, e.g. 'Server IOR Python'.
        # These keys can be checked for, and if absent the SIDL file would be compiled
        generatedSource = self.executeSection(self.buildSIDL, builder, f)
	if self.project.getUrl() == 'bk://ase.bkbits.net/Runtime':
          for language in self.serverLanguages:
            self.executeSection(getattr(self, 'build'+language+'Server'), builder, f, language, generatedSource)
        for language in self.clientLanguages:
          self.executeSection(getattr(self, 'build'+language+'Client'), builder, f, language, generatedSource)
	if not self.project.getUrl() == 'bk://ase.bkbits.net/Runtime':
          for language in self.serverLanguages:
            self.executeSection(getattr(self, 'build'+language+'Server'), builder, f, language, generatedSource)
        self.argDB.save(force = 1)
        shutil.copy(self.argDB.saveFilename, self.argDB.saveFilename+'.bkp')
        builder.sourceDB.save()
        shutil.copy(str(builder.sourceDB.filename), str(builder.sourceDB.filename)+'.bkp')
    return

  def install(self, builder, argDB):
    '''Install all necessary data for this project into the current RDict
       - FIX: Build project graph
       - FIX: Update language specific information'''
    if not 'installedprojects' in argDB:
      return
    for sidlFile in self.sidl:
      baseName = os.path.splitext(os.path.basename(sidlFile))[0]
      #self.loadConfiguration(builder, 'SIDL '+baseName)
      for language in self.serverLanguages:
        self.project.appendPath(language, os.path.join(self.root, self.getSIDLServerDirectory(builder, sidlFile, language)))
      for language in self.clientLanguages:
        self.project.appendPath(language, os.path.join(self.root, self.getSIDLClientDirectory(builder, sidlFile, language)))
    # self.compileTemplate.install()
    projects = filter(lambda project: not project.getUrl() == self.project.getUrl(), argDB['installedprojects'])
    argDB['installedprojects'] = projects+[self.project]
    self.logPrint('Installed project '+str(self.project), debugSection = 'install')
    # Update project in 'projectDependenceGraph'
    return
