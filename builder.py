import logging
import script

class CompileError(RuntimeError):
  pass

class LinkError(RuntimeError):
  pass

class CompilerDependencyChecker(logging.Logger):
  def __init__(self):
    import sourceDatabase
    import os

    logging.Logger.__init__(self)
    self.sourceDB = sourceDatabase.SourceDB(os.getcwd())
    self.sourceDB.load()
    return

  def __call__(self, source, target):
    '''This method determines whether source should be recompiled into target
       - It checks that source exists
       - If target is not None and does not exist, rebuild
       - If source is not in the databasem rebuild
       - If the checksum for source has changed, rebuild
       - If any dependency would be rebuilt, rebuild'''
    import os

    for f in source:
      if not os.path.isfile(f):
        raise CompileError('Source file not found for compile: '+str(f))
    if not target is None and not os.path.isfile(target):
      self.logPrint('Source '+str(source)+' rebuilds due to missing target '+str(target))
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

  def update(self, source):
    [self.sourceDB.updateSource(f) for f in source]
    self.sourceDB.save()
    return

class Builder(logging.Logger):
  def __init__(self):
    logging.Logger.__init__(self)
    self.language          = []
    self.configurations    = {}
    self.configurationName = []
    self.pushConfiguration('default')
    self.shouldCompile     = CompilerDependencyChecker()
    return

  def setup(self):
    logging.Logger.setup(self)
    self.getLanguageProcessor().setup()
    self.shouldCompile.setup()
    return

  def pushLanguage(self, language):
    '''Set the current language'''
    self.language.append(language)
    return self.language[-1]

  def popLanguage(self):
    '''Restore the previous language'''
    self.language.pop()
    return self.language[-1]

  def getConfiguration(self, configurationName):
    if not configurationName in self.configurations:
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
    import os

    base, ext = os.path.splitext(source)
    return base+'.o'

  def getCompilerCommand(self, source, target = None):
    self.getCompiler()
    if target is None:
      target = self.getCompilerTarget(source[0])
    return self.getLanguageProcessor().getCompilerObject(self.language[-1]).getCommand(source, target)

  def compile(self, source, target = None):
    '''Return the error output from this compile and the return code'''
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

    if target is None:
      target = self.getCompilerTarget(source[0])
    if self.shouldCompile(source, target):
      return script.Script.executeShellCommand(self.getCompilerCommand(source, target), checkCommand = check, log = self.log)
    return ('', '', 0)

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
    import os
    import sys

    base, ext = os.path.splitext(source)
    if shared:
      return base+'.so'
    if sys.platform[:3] == 'win' or sys.platform == 'cygwin':
      return base+'.exe'
    return base

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
        if len(filter(lambda l: l.find('warning') < 0, error.split('\n'))):
          raise LinkError(output+error)
      return

    (out, err, ret) = script.Script.executeShellCommand(self.getLinkerCommand(source, target, shared), checkCommand = check, log = self.log)
    return (out+err, ret)

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
