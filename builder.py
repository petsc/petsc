import logging
import script

class CompileError(RuntimeError):
  pass

class LinkError(RuntimeError):
  pass

class DependencyChecker(logging.Logger):
  '''This class is a template for checking dependencies between sources and targets, and among sources'''
  sourceDB = None

  def __init__(self):
    logging.Logger.__init__(self)
    if DependencyChecker.sourceDB is None:
      import sourceDatabase
      import os

      DependencyChecker.sourceDB = sourceDatabase.SourceDB(os.getcwd())
    self.sourceDB = DependencyChecker.sourceDB
    self.sourceDB.load()
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
  def __init__(self):
    logging.Logger.__init__(self)
    self.language          = []
    self.configurations    = {}
    self.configurationName = []
    self.shouldCompile     = MD5DependencyChecker()
    self.shouldLink        = TimeDependencyChecker()
    self.pushConfiguration('default')
    return

  def setup(self):
    logging.Logger.setup(self)
    self.getLanguageProcessor().setup()
    self.shouldCompile.setup()
    self.shouldLink.setup()
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
      self.configurations[configurationName] = script.LanguageProcessor()
      self.configurations[configurationName].setup()
    return self.configurations[configurationName]

  def pushConfiguration(self, configurationName):
    '''Set the current configuration'''
    self.configurationName.append(configurationName)
    return self.getConfiguration(self.configurationName[-1])

  def popConfiguration(self):
    '''Restore the previous configuration'''
    self.configurationName.pop()
    return self.getConfiguration(self.configurationName[-1])

  def saveConfiguration(self, configurationName):
    '''Save a configuration to RDict'''
    import cPickle

    self.argDB['#'+configurationName+' cache#'] = cPickle.dumps(self.getConfiguration(configurationName))
    return

  def loadConfiguration(self, configurationName):
    '''Load a configuration from RDict'''
    loadName = '#'+configurationName+' cache#'
    if loadName in self.argDB:
      import cPickle

      self.configurations[configurationName] = cPickle.loads(self.argDB[loadName])
    return self.getConfiguration(configurationName)

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
        self.framework.log.write('Possible ERROR while running preprocessor: '+error)
        if status: self.framework.log.write('ret = '+str(status)+'\n')
        if error: self.framework.log.write('error message = {'+error+'}\n')
        self.framework.log.write('Source:\n'+self.getCode(codeStr))
      return

    command = self.getPreprocessorCmd()
    self.framework.outputHeader(self.compilerDefines)
    f = file(self.compilerSource, 'w')
    f.write(self.getCode(codeStr))
    f.close()
    (out, err, ret) = script.Script.executeShellCommand(command, checkCommand = report, log = self.framework.log)
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
          outputFiles = {'ELF': [target]}
        else:
          outputFiles = {}
      config.outputFiles.update(outputFiles)
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
      obj.pushRequiredFlags(self.argDB['SHARED_LIBRARY_FLAG'])
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
      target = self.getLinkerTarget(source[0], shared)
    if self.shouldLink(source, target):
      if callable(self.getLinkerObject()):
        output, error, status, outputFiles = self.getLinkerObject()(source, target)
        check(None, status, output, error)
      else:
        output, error, status = script.Script.executeShellCommand(self.getLinkerCommand(source, target, shared), checkCommand = check, log = self.log)
        outputFiles = {'Linked ELF': [target]}
      config.outputFiles.update(outputFiles)
    else:
      output      = ''
      error       = ''
      status      = 0
      outputFiles = {}
    return (output, error, status, outputFiles)

  def run(self, includes, body, cleanup = 1, defaultOutputArg = ''):
    if not self.checkLink(includes, body, cleanup = 0): return ('', 1)
    if not os.path.isfile(self.linkerObj) or not os.access(self.linkerObj, os.X_OK):
      self.framework.log.write('ERROR while running executable: '+self.linkerObj+' is not executable')
      return ('', 1)
    if not self.framework.argDB['can-execute']:
      if defaultOutputArg:
        if defaultOutputArg in self.framework.argDB:
          return (self.framework.argDB[defaultOutputArg], 0)
        else:
          raise RuntimeError('Must give a default value for '+defaultOutputArg+' since executables cannot be run')
      else:
        raise RuntimeError('Running executables on this system is not supported')
    command = './'+self.linkerObj
    output  = ''
    error   = ''
    status  = 1
    self.framework.log.write('Executing: '+command+'\n')
    try:
      (output, error, status) = script.Script.executeShellCommand(command, log = self.framework.log)
    except RuntimeError, e:
      self.framework.log.write('ERROR while running executable: '+str(e)+'\n')
    if os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    if cleanup and os.path.isfile(self.linkerObj): os.remove(self.linkerObj)
    return (output+error, status)
