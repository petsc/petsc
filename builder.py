import logging
import script

import sets

class CompileError(RuntimeError):
  pass

class LinkError(RuntimeError):
  pass

class DependencyChecker(logging.Logger):
  '''This class is a template for checking dependencies between sources and targets, and among sources'''
  def __init__(self, sourceDB, clArgs = None, argDB = None):
    logging.Logger.__init__(self, clArgs, argDB)
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
      return True
    return False

  def update(self, source):
    [self.sourceDB.updateSource(f) for f in source]
    self.sourceDB.save()
    return

class MD5DependencyChecker(DependencyChecker):
  '''This class uses MD5 fingerprints and a database to detect changes in files'''
  def __call__(self, source, target):
    '''This method determines whether source should be recompiled into target
       - If the superclass returns True, then rebuild
       - If source is not in the database, then rebuild
       - If the checksum for source has changed, then rebuild
       - If any dependency would be rebuilt, then rebuild'''
    if DependencyChecker.__call__(self, source, target):
      return True
    for f in source:
      if not f in self.sourceDB:
        self.logPrint('Source '+str(source)+' rebuilds due to file '+str(f)+' missing from database')
        return True
      if not self.sourceDB[f][0] == self.sourceDB.getChecksum(f):
        self.logPrint('Source '+str(source)+' rebuilds due to changed checksum of file '+str(f))
        return True
      for dep in self.sourceDB[f][3]:
        if self([dep], None):
          self.logPrint('Source '+str(source)+' rebuilds due to rebuilt dependecy '+str(dep))
          return True
    self.logPrint('Source '+str(source)+' will not be rebuilt into target '+str(target))
    return False

class TimeDependencyChecker(DependencyChecker):
  '''This class uses modification times to detect changes in files'''
  def __call__(self, source, target):
    '''This method determines whether source should be recompiled into target
       - If the superclass returns True, then rebuild
       - If source is not in the database, then rebuild
       - If the checksum for source has changed, then rebuild
       - If any dependency would be rebuilt, then rebuild'''
    if DependencyChecker.__call__(self, source, target):
      return True
    if not target is None:
      targetModTime = self.sourceDB.getModificationTime(target)
    for f in source:
      if not f in self.sourceDB:
        self.logPrint('Source '+str(source)+' rebuilds due to file '+str(f)+' missing from database')
        return True
      if self.sourceDB[f][1] < self.sourceDB.getModificationTime(f):
        self.logPrint('Source '+str(source)+' rebuilds due to changed modification time of file '+str(f))
        return True
      if targetModTime < self.sourceDB.getModificationTime(f):
        self.logPrint('Source '+str(source)+' rebuilds due to later modification time of '+str(f)+' than target '+str(target))
        return True
      for dep in self.sourceDB[f][3]:
        if self([dep], None):
          self.logPrint('Source '+str(source)+' rebuilds due to rebuilt dependecy '+str(dep))
          return True
    self.logPrint('Source '+str(source)+' will not be rebuilt into target '+str(target))
    return False

  def update(self, source):
    '''Do not calculate a checksum, as it may be too expensive'''
    [self.sourceDB.updateSource(f, noChecksum = 1) for f in source]
    self.sourceDB.save()
    return

class Builder(logging.Logger):
  def __init__(self, framework, sourceDB = None):
    import sourceControl
    import sourceDatabase

    logging.Logger.__init__(self, argDB = framework.argDB)
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

  def getFramework(self):
    return self._framework
  def setFramework(self, framework):
    self._framework   = framework
    self.setCompilers = framework.require('config.setCompilers', None)
    self.compilers    = framework.require('config.compilers', None)
    return
  framework = property(getFramework, setFramework, doc = 'The configure framework')

  def setup(self):
    logging.Logger.setup(self)
    self.getLanguageProcessor().setup()
    self.shouldCompile.setup()
    self.shouldLink.setup()
    self.versionControl.setup()
    return

  def pushLanguage(self, language):
    '''Set the current language'''
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
      self.configurations[configurationName] = script.LanguageProcessor(argDB = self.argDB, compilers = self.compilers)
      self.configurations[configurationName].setup()
    return self.configurations[configurationName]

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
    import cPickle

    cache = cPickle.dumps(self.getConfiguration(configurationName))
    self.argDB['#'+configurationName+' cache#'] = cache
    self.logPrint('Wrote configuration '+configurationName+' to cache: size '+str(len(cache)))
    return

  def loadConfiguration(self, configurationName):
    '''Load a configuration from RDict'''
    loadName = '#'+configurationName+' cache#'
    if loadName in self.argDB:
      import cPickle

      try:
        cache = self.argDB[loadName]
        self.configurations[configurationName]       = cPickle.loads(cache)
        self.configurations[configurationName].argDB = self.argDB
        self.logPrint('Loaded configuration '+configurationName+' from cache: size '+str(len(cache)))
      except ValueError, e:
        if str(e) == 'insecure string pickle':
          del self.argDB[loadName]
        else:
          raise e
    return self.getConfiguration(configurationName)

  def updateOutputFiles(self, outputFiles, newOutputFiles):
    for language in newOutputFiles:
      if language in outputFiles:
        if isinstance(outputFiles[language], sets.Set) and isinstance(outputFiles[language], sets.Set):
          outputFiles[language].union_update(newOutputFiles[language])
        elif isinstance(outputFiles[language], dict) and isinstance(outputFiles[language], dict):
          self.updateOutputFiles(outputFiles[language], outputFiles[language])
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
    return self.getLanguageProcessor().argDB[compiler.name]

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
        # This is a hack
        if len(''.join(filter(lambda l: l.find('warning') < 0 and l.find('In function') < 0, error.split('\n')))):
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
    return self.getLanguageProcessor().argDB[linker.name]

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

  def getLinkerCommand(self, source, target = None, shared = 0):
    self.getLinker()
    obj = self.getLanguageProcessor().getLinkerObject(self.language[-1])
    if target is None:
      target = self.getLinkerTarget(source[0], shared)
    if shared:
      obj.pushRequiredFlags(self.setCompilers.sharedLibraryFlag)
    command = obj.getCommand(source, target)
    if shared:
      obj.popRequiredFlags()
    return command

  def link(self, source, target = None, shared = 0):
    def check(command, status, output, error):
      if error or status:
        self.logWrite('Possible ERROR while running linker: '+output)
        if status: self.logWrite('ret = '+str(status)+'\n')
        if error: self.logWrite('error message = {'+error+'}\n')
        self.logWrite('Source:\n'+str(source)+'\n')
        # This is a hack
        if len(filter(lambda l: l.find('warning') < 0, error.split('\n'))):
          raise LinkError(output+error)
      self.shouldLink.update(source)
      return

    config = self.getConfiguration()
    if target is None:
      if len(source) and not source[0] is None:
        target = self.getLinkerTarget(source[0], shared)
    if not target is None and self.shouldLink(source, target):
        if callable(self.getLinkerObject()):
          output, error, status, outputFiles = self.getLinkerObject()(source, target)
          check(None, status, output, error)
        else:
          output, error, status = script.Script.executeShellCommand(self.getLinkerCommand(source, target, shared), checkCommand = check, log = self.log)
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
    if not self.argDB['can-execute']:
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
