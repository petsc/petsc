import logger
import script

try:
  from types import ModuleType
  sets = ModuleType('sets')
  sets.Set = set
  sets.ImmutableSet = frozenset
except NameError:
  try:
    import sets
  except ImportError:
    import config.setsBackport as sets

import cPickle

class CompileError(RuntimeError):
  pass

class LinkError(RuntimeError):
  pass

class DependencyChecker(logger.Logger):
  '''This class is a template for checking dependencies between sources and targets, and among sources'''
  def __init__(self, sourceDB, clArgs = None, argDB = None):
    logger.Logger.__init__(self, clArgs, argDB)
    self.sourceDB = sourceDB
    return

  def __call__(self, source, target):
    '''This method determines whether source should be recompiled into target
       - It checks that source exists
       - If target is not None and does not exist, rebuild'''
    import os

    for f in source:
      if not os.path.isfile(f):
        raise RuntimeError('Source file not found for compile: '+str(f))
    if not target is None and not os.path.isfile(target):
      self.logPrint('Source '+str(source)+' rebuilds due to missing target '+str(target))
      return 1
    return 0

  def update(self, source):
    '''Update the information for source in the database'''
    [self.sourceDB.updateSource(f) for f in source]
    self.sourceDB.save()
    self.logPrint('Updated '+str(source)+' in source database')
    return

  def force(self, source):
    '''Remove the information for source from the database, forcing a rebuild'''
    [self.sourceDB.clearSource(f) for f in source]
    self.sourceDB.save()
    self.logPrint('Removed information about '+str(source)+' from source database')
    return

class MD5DependencyChecker(DependencyChecker):
  '''This class uses MD5 fingerprints and a database to detect changes in files'''
  def __call__(self, source, target, checked = None):
    '''This method determines whether source should be recompiled into target
       - If the superclass returns True, then rebuild
       - If source is not in the database, then rebuild
       - If the checksum for source has changed, then rebuild
       - If any dependency would be rebuilt, then rebuild'''
    if DependencyChecker.__call__(self, source, target):
      return 1
    for f in source:
      if not f in self.sourceDB:
        self.logPrint('Source '+str(source)+' rebuilds due to file '+str(f)+' missing from database')
        return 1
      checksum = self.sourceDB.getChecksum(f)
      if not self.sourceDB[f][0] == checksum:
        self.logPrint('Source '+str(source)+' rebuilds due to changed checksum('+str(checksum)+') of file '+str(f))
        return 1
      if checked is None:
        checked = sets.Set()
      for dep in self.sourceDB[f][3]:
        if dep in checked:
          continue
        else:
          checked.add(dep)
        if self([dep], None, checked):
          self.logPrint('Source '+str(source)+' rebuilds due to rebuilt dependecy '+str(dep))
          return 1
    self.logPrint('Source '+str(source)+' will not be rebuilt into target '+str(target))
    return 0

class TimeDependencyChecker(DependencyChecker):
  '''This class uses modification times to detect changes in files'''
  def __call__(self, source, target, checked = None):
    '''This method determines whether source should be recompiled into target
       - If the superclass returns True, then rebuild
       - If source is not in the database, then rebuild
       - If the checksum for source has changed, then rebuild
       - If any dependency would be rebuilt, then rebuild'''
    if DependencyChecker.__call__(self, source, target):
      return 1
    if not target is None:
      targetModTime = self.sourceDB.getModificationTime(target)
    for f in source:
      if not f in self.sourceDB:
        self.logPrint('Source '+str(source)+' rebuilds due to file '+str(f)+' missing from database')
        return 1
      if self.sourceDB[f][1] < self.sourceDB.getModificationTime(f):
        self.logPrint('Source '+str(source)+' rebuilds due to changed modification time of file '+str(f))
        return 1
      if targetModTime < self.sourceDB.getModificationTime(f):
        self.logPrint('Source '+str(source)+' rebuilds due to later modification time of '+str(f)+' than target '+str(target))
        return 1
      if checked is None:
        checked = sets.Set()
      for dep in self.sourceDB[f][3]:
        if dep in checked:
          continue
        else:
          checked.add(dep)
        if self([dep], None, checked):
          self.logPrint('Source '+str(source)+' rebuilds due to rebuilt dependecy '+str(dep))
          return 1
    self.logPrint('Source '+str(source)+' will not be rebuilt into target '+str(target))
    return 0

  def update(self, source):
    '''Do not calculate a checksum, as it may be too expensive'''
    [self.sourceDB.updateSource(f, noChecksum = 1) for f in source]
    self.sourceDB.save()
    self.logPrint('Updated '+str(source)+' in source database')
    return

class Builder(logger.Logger):
  def __init__(self, framework, sourceDB = None):
    import sourceControl
    import sourceDatabase

    logger.Logger.__init__(self, argDB = framework.argDB)
    self.framework         = framework
    self.language          = []
    self.configurations    = {}
    self.configurationName = []
    if sourceDB is None:
      self.sourceDB        = sourceDatabase.SourceDB(self.root)
    else:
      self.sourceDB        = sourceDB
    self.shouldCompile     = MD5DependencyChecker(self.sourceDB, argDB = self.argDB)
    self.shouldLink        = TimeDependencyChecker(self.sourceDB, argDB = self.argDB)
    self.versionControl    = sourceControl.BitKeeper(argDB = self.argDB)
    self.sourceDB.load()
    self.pushConfiguration('default')
    return

  def getSetCompilers(self):
    return self.framework.require('config.setCompilers', None)
  setCompilers = property(getSetCompilers, doc = 'The config.setCompilers configure object')
  def getCompilers(self):
    return self.framework.require('config.compilers', None)
  compilers = property(getCompilers, doc = 'The config.compilers configure object')
  def getLibraries(self):
    return self.framework.require('config.libraries', None)
  libraries = property(getLibraries, doc = 'The config.libraries configure object')

  def setup(self):
    logger.Logger.setup(self)
    self.getLanguageProcessor().setup()
    self.shouldCompile.setup()
    self.shouldLink.setup()
    self.versionControl.setup()
    return

  def pushLanguage(self, language):
    '''Set the current language'''
    if language == 'C++': language = 'Cxx'
    self.language.append(language)
    return self.language[-1]

  def popLanguage(self):
    '''Restore the previous language'''
    self.language.pop()
    return self.language[-1]

  def getConfiguration(self, configurationName = None):
    '''Retrieve the configuration with the given name
       - Create a new one if none exists
       - If no name is given, return the current configuration'''
    if configurationName is None:
      configurationName = self.configurationName[-1]
    elif not configurationName in self.configurations:
      self.configurations[configurationName] = script.LanguageProcessor(argDB = self.argDB, framework = self.framework, versionControl = self.versionControl)
      self.configurations[configurationName].setup()
      for language in self.framework.preprocessorObject:
        self.framework.getPreprocessorObject(language).copy(self.configurations[configurationName].getPreprocessorObject(language))
      for language in self.framework.compilerObject:
        self.framework.getCompilerObject(language).copy(self.configurations[configurationName].getCompilerObject(language))
      for language in self.framework.linkerObject:
        self.framework.getLinkerObject(language).copy(self.configurations[configurationName].getLinkerObject(language))
      for language in self.framework.sharedLinkerObject:
        oldObj = self.framework.getSharedLinkerObject(language)
        newObj = oldObj.__class__(oldObj.argDB)
        newObj.copy(oldObj)
        self.configurations[configurationName].setSharedLinkerObject(language, newObj)
    configuration = self.configurations[configurationName]
    configuration.framework = self.framework
    configuration.versionControl = self.versionControl
    return configuration

  def pushConfiguration(self, configurationName):
    '''Set the current configuration'''
    self.logPrint('Pushed configuration '+configurationName, debugSection = 'build')
    self.configurationName.append(configurationName)
    return self.getConfiguration(self.configurationName[-1])

  def popConfiguration(self):
    '''Restore the previous configuration'''
    configurationName = self.configurationName.pop()
    self.logPrint('Popped configuration '+configurationName, debugSection = 'build')
    return self.getConfiguration(self.configurationName[-1])

  def saveConfiguration(self, configurationName):
    '''Save a configuration to RDict'''
    cache = cPickle.dumps(self.getConfiguration(configurationName))
    self.argDB['#'+configurationName+' cache#'] = cache
    self.logPrint('Wrote configuration '+configurationName+' to cache: size '+str(len(cache))+' in '+self.argDB.saveFilename)
    return

  def loadConfiguration(self, configurationName):
    '''Load a configuration from RDict'''
    loadName = '#'+configurationName+' cache#'
    if loadName in self.argDB:

      try:
        cache = self.argDB[loadName]
        self.configurations[configurationName] = cPickle.loads(cache)
        self.configurations[configurationName].framework = self.framework
        self.configurations[configurationName].argDB = self.argDB
        self.logPrint('Loaded configuration '+configurationName+' from cache: size '+str(len(cache))+' in '+self.argDB.saveFilename)
      except ValueError, e:
        if str(e) == 'insecure string pickle':
          del self.argDB[loadName]
        else:
          raise e
      except cPickle.BadPickleGet:
        del self.argDB[loadName]
    return self.getConfiguration(configurationName)

  def updateOutputFiles(self, outputFiles, newOutputFiles):
    '''Update current output file sets with new file sets
       - If the current language is SIDL, outputFiles is first cleared'''
    if self.language[-1] == 'SIDL':
      outputFiles.clear()
    for language in newOutputFiles:
      if language in outputFiles:
        if isinstance(outputFiles[language], sets.Set) and isinstance(newOutputFiles[language], sets.Set):
          outputFiles[language].update(newOutputFiles[language])
        elif isinstance(outputFiles[language], dict) and isinstance(newOutputFiles[language], dict):
          self.updateOutputFiles(outputFiles[language], newOutputFiles[language])
        else:
          raise RuntimeError('Mismatched output files')
      else:
        outputFiles[language] = newOutputFiles[language]
    return outputFiles

  def getLanguageProcessor(self):
    return self.configurations[self.configurationName[-1]]

  def getPreprocessor(self):
    preprocessor = self.getLanguageProcessor().getPreprocessorObject(self.language[-1])
    preprocessor.checkSetup()
    return self.getLanguageProcessor().argDB[preprocessor.name]

  def getPreprocessorFlags(self):
    return self.getLanguageProcessor().getPreprocessorObject(self.language[-1]).getFlags()

  def getPreprocessorCommand(self):
    self.getPreprocessor()
    return self.getLanguageProcessor().getPreprocessorObject(self.language[-1]).getCommand(self.compilerSource)

  def preprocess(self, codeStr):
    def report(command, status, output, error):
      if error or status:
        self.logWrite('Possible ERROR while running preprocessor: '+error)
        if status: self.logWrite('ret = '+str(status)+'\n')
        if error: self.logWrite('error message = {'+error+'}\n')
        self.logWrite('Source:\n'+self.getCode(codeStr))
      return

    command = self.getPreprocessorCmd()
    self.framework.outputHeader(self.compilerDefines)
    f = file(self.compilerSource, 'w')
    f.write(self.getCode(codeStr))
    f.close()
    (out, err, ret) = script.Script.executeShellCommand(command, checkCommand = report, log = self.log)
    if os.path.isfile(self.compilerDefines): os.remove(self.compilerDefines)
    if os.path.isfile(self.compilerSource): os.remove(self.compilerSource)
    return (out, err, ret)

  def getCompiler(self):
    compiler = self.getLanguageProcessor().getCompilerObject(self.language[-1])
    compiler.checkSetup()
    return compiler.getProcessor()

  def getCompilerFlags(self):
    return self.getLanguageProcessor().getCompilerObject(self.language[-1]).getFlags()

  def setCompilerFlags(self, flags):
    return self.getLanguageProcessor().getCompilerObject(self.language[-1]).setFlags(flags)

  def getCompilerTarget(self, source):
    return self.getLanguageProcessor().getCompilerObject(self.language[-1]).getTarget(source)

  def getCompilerObject(self):
    compiler = self.getLanguageProcessor().getCompilerObject(self.language[-1])
    compiler.checkSetup()
    return compiler

  def getCompilerCommand(self, source, target = None):
    self.getCompiler()
    if target is None:
      target = self.getCompilerTarget(source[0])
    return self.getLanguageProcessor().getCompilerObject(self.language[-1]).getCommand(source, target)

  def compile(self, source, target = None):
    '''Compile the list of source files into target
       - Return the standard output, error output, return code, and a dictionary mapping languages to lists of output files
       - This method checks whether the compile should occur'''
    def check(command, status, output, error):
      if error or status:
        self.logWrite('Possible ERROR while running compiler: '+output)
        if status: self.logWrite('ret = '+str(status)+'\n')
        if error: self.logWrite('error message = {'+error+'}\n')
        self.logWrite('Source:\n'+str(source)+'\n')
        if not self.argDB['ignoreCompileOutput']:
          # This is a hack
          if len(''.join(filter(lambda l: l.find('warning') < 0 and l.find('In function') < 0 and l.find('In member function') < 0 and l.find('At top level') < 0 and l.find('In file included from') < 0 and not l.strip().startswith('from '), error.split('\n')))):
            raise CompileError(output+error)
      self.shouldCompile.update(source)
      return

    config = self.getConfiguration()
    if target is None:
      target = self.getCompilerTarget(source[0])
    if self.shouldCompile(source, target):
      if callable(self.getCompilerObject()):
        output, error, status, outputFiles = self.getCompilerObject()(source, target)
        check(None, status, output, error)
      else:
        output, error, status = script.Script.executeShellCommand(self.getCompilerCommand(source, target), checkCommand = check, log = self.log)
        if not target is None:
          outputFiles = {'ELF': sets.Set([target])}
        else:
          outputFiles = {}
      self.updateOutputFiles(config.outputFiles, outputFiles)
    else:
      output      = ''
      error       = ''
      status      = 0
      outputFiles = {}
    return (output, error, status, outputFiles)

  def getLinker(self):
    linker = self.getLanguageProcessor().getLinkerObject(self.language[-1])
    linker.checkSetup()
    return linker.getProcessor()

  def getLinkerFlags(self):
    return self.getLanguageProcessor().getLinkerObject(self.language[-1]).getFlags()

  def setLinkerFlags(self, flags):
    return self.getLanguageProcessor().getLinkerObject(self.language[-1]).setFlags(flags)

  def getLinkerExtraArguments(self):
    return self.getLanguageProcessor().getLinkerObject(self.language[-1]).getExtraArguments()

  def setLinkerExtraArguments(self, args):
    return self.getLanguageProcessor().getLinkerObject(self.language[-1]).setExtraArguments(args)

  def getLinkerTarget(self, source, shared):
    return self.getLanguageProcessor().getLinkerObject(self.language[-1]).getTarget(source, shared)

  def getLinkerObject(self):
    compiler = self.getLanguageProcessor().getLinkerObject(self.language[-1])
    compiler.checkSetup()
    return compiler

  def getLinkerCommand(self, source, target = None):
    self.getLinker()
    obj = self.getLanguageProcessor().getLinkerObject(self.language[-1])
    if target is None:
      target = self.getLinkerTarget(source[0], shared)
    command = obj.getCommand(source, target)
    return command

  def getSharedLinker(self):
    linker = self.getLanguageProcessor().getSharedLinkerObject(self.language[-1])
    linker.checkSetup()
    return linker.getProcessor()

  def getSharedLinkerFlags(self):
    return self.getLanguageProcessor().getSharedLinkerObject(self.language[-1]).getFlags()

  def setSharedLinkerFlags(self, flags):
    return self.getLanguageProcessor().getSharedLinkerObject(self.language[-1]).setFlags(flags)

  def getSharedLinkerExtraArguments(self):
    return self.getLanguageProcessor().getSharedLinkerObject(self.language[-1]).getExtraArguments()

  def setSharedLinkerExtraArguments(self, args):
    return self.getLanguageProcessor().getSharedLinkerObject(self.language[-1]).setExtraArguments(args)

  def getSharedLinkerTarget(self, source, shared, prefix = 'lib'):
    return self.getLanguageProcessor().getSharedLinkerObject(self.language[-1]).getTarget(source, shared, prefix)

  def getSharedLinkerObject(self):
    compiler = self.getLanguageProcessor().getSharedLinkerObject(self.language[-1])
    compiler.checkSetup()
    return compiler

  def getSharedLinkerCommand(self, source, target = None):
    self.getLinker()
    obj = self.getLanguageProcessor().getSharedLinkerObject(self.language[-1])
    if target is None:
      target = self.getLinkerTarget(source[0], shared = 1)
    command = obj.getCommand(source, target)
    return command

  def getDynamicLinker(self):
    linker = self.getLanguageProcessor().getDynamicLinkerObject(self.language[-1])
    linker.checkSetup()
    return linker.getProcessor()

  def getDynamicLinkerFlags(self):
    return self.getLanguageProcessor().getDynamicLinkerObject(self.language[-1]).getFlags()

  def setDynamicLinkerFlags(self, flags):
    return self.getLanguageProcessor().getDynamicLinkerObject(self.language[-1]).setFlags(flags)

  def getDynamicLinkerExtraArguments(self):
    return self.getLanguageProcessor().getDynamicLinkerObject(self.language[-1]).getExtraArguments()

  def setDynamicLinkerExtraArguments(self, args):
    return self.getLanguageProcessor().getDynamicLinkerObject(self.language[-1]).setExtraArguments(args)

  def getDynamicLinkerTarget(self, source, shared, prefix = 'lib'):
    return self.getLanguageProcessor().getDynamicLinkerObject(self.language[-1]).getTarget(source, shared, prefix)

  def getDynamicLinkerObject(self):
    compiler = self.getLanguageProcessor().getDynamicLinkerObject(self.language[-1])
    compiler.checkSetup()
    return compiler

  def getDynamicLinkerCommand(self, source, target = None):
    self.getLinker()
    obj = self.getLanguageProcessor().getDynamicLinkerObject(self.language[-1])
    if target is None:
      target = self.getLinkerTarget(source[0], shared = 1)
    command = obj.getCommand(source, target)
    return command

  def link(self, source, target = None, shared = 0):
    def check(command, status, output, error):
      if error or status:
        self.logWrite('Possible ERROR while running linker: '+output)
        if status: self.logWrite('ret = '+str(status)+'\n')
        if error: self.logWrite('error message = {'+error+'}\n')
        self.logWrite('Source:\n'+str(source)+'\n')
        self.logWrite('Command:\n'+str(command)+'\n')
        # This is a hack, but a damn important one
        if error:
          error = error.splitlines()
          error = filter(lambda l: l.find('warning') < 0, error)
          # Mac bundles always have undefined environ variable
          error = filter(lambda l: l.find('environ') < 0, error)
          if len(error):
            raise LinkError(output+'\n'.join(error))
      self.shouldLink.update(source)
      return

    config = self.getConfiguration()
    if target is None:
      if len(source) and not source[0] is None:
        if shared == 'dynamic':
          target = self.getDynamicLinkerTarget(source[0], shared)
        elif shared:
          target = self.getSharedLinkerTarget(source[0], shared)
        else:
          target = self.getLinkerTarget(source[0], shared)
    if not target is None and self.shouldLink(source, target):
        if callable(self.getLinkerObject()):
          if shared == 'dynamic':
            output, error, status, outputFiles = self.getDynamicLinkerObject()(source, target)
          elif shared:
            output, error, status, outputFiles = self.getSharedLinkerObject()(source, target)
          else:
            output, error, status, outputFiles = self.getLinkerObject()(source, target)
          check(None, status, output, error)
        else:
          if shared == 'dynamic':
            output, error, status = script.Script.executeShellCommand(self.getDynamicLinkerCommand(source, target), checkCommand = check, log = self.log)
          elif shared:
            output, error, status = script.Script.executeShellCommand(self.getSharedLinkerCommand(source, target), checkCommand = check, log = self.log)
          else:
            output, error, status = script.Script.executeShellCommand(self.getLinkerCommand(source, target), checkCommand = check, log = self.log)
          outputFiles = {'Linked ELF': sets.Set([target])}
        self.updateOutputFiles(config.outputFiles, outputFiles)
    else:
      output      = ''
      error       = ''
      status      = 0
      outputFiles = {}
    return (output, error, status, outputFiles)

  def run(self, includes, body, cleanup = 1, defaultOutputArg = ''):
    if not self.checkLink(includes, body, cleanup = 0): return ('', 1)
    if not os.path.isfile(self.linkerObj) or not os.access(self.linkerObj, os.X_OK):
      self.logWrite('ERROR while running executable: '+self.linkerObj+' is not executable')
      return ('', 1)
    if self.argDB['with-batch']:
      if defaultOutputArg:
        if defaultOutputArg in self.argDB:
          return (self.argDB[defaultOutputArg], 0)
        else:
          raise RuntimeError('Must give a default value for '+defaultOutputArg+' since executables cannot be run')
      else:
        raise RuntimeError('Running executables on this system is not supported')
    command = './'+self.linkerObj
    output  = ''
    error   = ''
    status  = 1
    self.logWrite('Executing: '+command+'\n')
    try:
      (output, error, status) = script.Script.executeShellCommand(command, log = self.log)
    except RuntimeError, e:
      self.logWrite('ERROR while running executable: '+str(e)+'\n')
    if os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    if cleanup and os.path.isfile(self.linkerObj): os.remove(self.linkerObj)
    return (output+error, status)
