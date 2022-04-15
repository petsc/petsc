from __future__ import absolute_import
import script

import os
import pickle
from functools import reduce

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
    except ImportError as e:
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
      cache = pickle.dumps(self.framework)
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
    self.logPrint('SECTION: '+str(section.__func__.__name__)+' in '+self.getRoot()+' from '+str(section.__self__.__class__.__module__)+'('+str(section.__func__.__code__.co_filename)+':'+str(section.__func__.__code__.co_firstlineno)+') at '+time.ctime(time.time()), debugSection = 'screen', indent = 0)
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
    except Exception as e:
      import sys, traceback
      self.logPrint('************************************ ERROR **************************************')
      traceback.print_tb(sys.exc_info()[2], file = self.log)
      raise
    return 1

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
        lib.includes = [inc for inc in lib.includes if inc]
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
        lib.includes = [inc for inc in lib.includes if inc]
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
        bin.includes = [inc for inc in bin.includes if inc]
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

