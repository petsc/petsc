from __future__ import print_function
from __future__ import absolute_import
import sys
if not hasattr(sys, 'version_info'):
  print('*** Python version 1 is not supported. Please get the latest version from www.python.org ***')
  sys.exit(4)

import pickle

import subprocess

import nargs

# Uses threads to monitor running programs and time them out if they take too long
useThreads = nargs.Arg.findArgument('useThreads', sys.argv[1:])
if useThreads == 'no' or useThreads == '0':
  useThreads = 0
elif useThreads == None or useThreads == 'yes' or useThreads == '1':
  useThreads = 1
else:
  raise RuntimeError('Unknown option value for --useThreads ',useThreads)

useSelect = nargs.Arg.findArgument('useSelect', sys.argv[1:])
if useSelect == 'no' or useSelect == '0':
  useSelect = 0
elif  useSelect is None or useSelect == 'yes' or useSelect == '1':
  useSelect = 1
else:
  raise RuntimeError('Unknown option value for --useSelect ',useSelect)

#  Run parts of configure in parallel, does not currently work;
#  see config/BuildSystem/config/framework.parallelQueueEvaluation()
useParallel = nargs.Arg.findArgument('useParallel', sys.argv[1:])
if useParallel == 'no' or useParallel == '0':
  useParallel = 0
elif  useParallel is None or useParallel == 'yes':
  useParallel = 5
else:
  if useParallel == '1':
    # handle case with --useParallel was used
    found = 0
    for i in sys.argv[1:]:
      if i.startswith('--useParallel='):
        found = 1
        break
    if found: useParallel = int(useParallel)
    else: useParallel = 5
useParallel = 0

import logger

class Script(logger.Logger):
  def __init__(self, clArgs = None, argDB = None, log = None):
    self.checkPython()
    logger.Logger.__init__(self, clArgs, argDB, log)
    self.shell = '/bin/sh'
    self.showHelp = 1
    return

  def hasHelpFlag(self):
    '''Decide whether to display the help message and exit'''
    import nargs

    if not self.showHelp:
      return 0
    if nargs.Arg.findArgument('help', self.clArgs) is None and nargs.Arg.findArgument('h', self.clArgs) is None:
      return 0
    return 1

  def hasListFlag(self):
    '''Decide whether to display the list of download files and exit'''
    import nargs

    if not self.showHelp:
      return 0
    if nargs.Arg.findArgument('with-packages-download-dir', self.clArgs) is None:
      return 0
    return 1

  def setupArguments(self, argDB):
    '''This method now also creates the help and action logs'''
    import help

    argDB = logger.Logger.setupArguments(self, argDB)

    self.help = help.Help(argDB)
    self.help.title = 'Script Help'

    self.actions = help.Info(argDB)
    self.actions.title = 'Script Actions'

    self.setupHelp(self.help)
    return argDB

  def setupHelp(self, help):
    '''This method should be overidden to provide help for arguments'''
    import nargs

    help.addArgument('Script', '-h',    nargs.ArgBool(None, 0, 'Print this help message', isTemporary = 1), ignoreDuplicates = 1)
    help.addArgument('Script', '-help', nargs.ArgBool(None, 0, 'Print this help message', isTemporary = 1), ignoreDuplicates = 1)
    help.addArgument('Script', '-with-packages-download-dir=<dir>', nargs.ArgDir(None,None, 'Skip network download of package tarballs and locate them in specified dir. If not found in dir, print package URL - so it can be obtained manually.', isTemporary = 1), ignoreDuplicates = 1)
    return help

  def setup(self):
    ''' This method checks to see whether help was requested'''
    if hasattr(self, '_setup'):
      return
    logger.Logger.setup(self)
    self._setup = 1
    if self.hasHelpFlag():
      self.argDB.readonly = True
      if self.argDB.target == ['default']:
        sections = None
      else:
        sections = self.argDB.target
      self.help.output(sections = sections)
      sys.exit()
    if self.hasListFlag():
      self.help.outputDownload()
    return

  def cleanup(self):
    '''This method outputs the action log'''
    self.actions.output(self.log)
    return

  def checkPython(self):
    if not hasattr(sys, 'version_info') or sys.version_info < (2,6):
      raise RuntimeError('BuildSystem requires Python version 2.6 or higher. Get Python at https://www.python.org/')
    return

  def getModule(root, name):
    '''Retrieve a specific module from the directory root, bypassing the usual paths'''
    import imp

    (fp, pathname, description) = imp.find_module(name, [root])
    try:
      return imp.load_module(name, fp, pathname, description)
    finally:
      if fp: fp.close()
    return
  getModule = staticmethod(getModule)

  def importModule(moduleName):
    '''Import the named module, and return the module object
       - Works properly for fully qualified names'''
    module     = __import__(moduleName)
    components = moduleName.split('.')
    for comp in components[1:]:
      module = getattr(module, comp)
    return module
  importModule = staticmethod(importModule)

  @staticmethod
  def runShellCommand(command, log=None, cwd=None):
    return Script.runShellCommandSeq([command], log=log, cwd=cwd)

  @staticmethod
  def runShellCommandSeq(commandseq, log=None, cwd=None):
    Popen = subprocess.Popen
    PIPE  = subprocess.PIPE
    output = ''
    error = ''
    for command in commandseq:
      useShell = isinstance(command, str) or isinstance(command, bytes)
      if log: log.write('Executing: %s\n' % (command,))
      try:
        pipe = Popen(command, cwd=cwd, stdin=None, stdout=PIPE, stderr=PIPE,
                     shell=useShell)
        (out, err) = pipe.communicate()
        if sys.version_info >= (3,0):
          out = out.decode(encoding='UTF-8',errors='replace')
          err = err.decode(encoding='UTF-8',errors='replace')
        ret = pipe.returncode
      except Exception as e:
        if hasattr(e,'message') and hasattr(e,'errno'):
          return ('', e.message, e.errno)
        else:
          return ('', str(e),1)
      output += out
      error += err
      if ret:
        break
    return (output, error, ret)

  def defaultCheckCommand(command, status, output, error):
    '''Raise an error if the exit status is nonzero'''
    if status: raise RuntimeError('Could not execute "%s":\n%s' % (command,output+error))
  defaultCheckCommand = staticmethod(defaultCheckCommand)

  def passCheckCommand(command, status, output, error):
    '''Does not check the command results'''
  passCheckCommand = staticmethod(passCheckCommand)

  @staticmethod
  def executeShellCommand(command, checkCommand = None, timeout = 600.0, log = None, lineLimit = 0, cwd=None, logOutputflg = True, threads = 0):
    '''Execute a shell command returning the output, and optionally provide a custom error checker
       - This returns a tuple of the (output, error, statuscode)'''
    '''The timeout is ignored unless the threads values is nonzero'''
    return Script.executeShellCommandSeq([command], checkCommand=checkCommand, timeout=timeout, log=log, lineLimit=lineLimit, cwd=cwd,logOutputflg = logOutputflg, threads = threads)

  @staticmethod
  def executeShellCommandSeq(commandseq, checkCommand = None, timeout = 600.0, log = None, lineLimit = 0, cwd=None, logOutputflg = True, threads = 0):
    '''Execute a sequence of shell commands (an && chain) returning the output, and optionally provide a custom error checker
       - This returns a tuple of the (output, error, statuscode)'''
    if not checkCommand:
      checkCommand = Script.defaultCheckCommand
    if log is None:
      log = logger.Logger.defaultLog
    def logOutput(log, output, logOutputflg):
      import re
      if not logOutputflg: return output
      # get rid of multiple blank lines
      output = re.sub('\n+','\n', output).strip()
      if output:
        if lineLimit:
          output = '\n'.join(output.split('\n')[:lineLimit])
        if '\n' in output:      # multi-line output
          log.write('stdout:\n'+output+'\n')
        else:
          log.write('stdout: '+output+'\n')
      return output
    def runInShell(commandseq, log, cwd):
      if useThreads and threads:
        import threading
        log.write('Running Executable with threads to time it out at '+str(timeout)+'\n')
        class InShell(threading.Thread):
          def __init__(self):
            threading.Thread.__init__(self)
            self.name = 'Shell Command'
            self.setDaemon(1)
          def run(self):
            (self.output, self.error, self.status) = ('', '', -1) # So these fields exist even if command fails with no output
            (self.output, self.error, self.status) = Script.runShellCommandSeq(commandseq, log, cwd)
        thread = InShell()
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
          error = 'Runaway process exceeded time limit of '+str(timeout)+'\n'
          log.write(error)
          return ('', error, -1)
        else:
          return (thread.output, thread.error, thread.status)
      else:
        return Script.runShellCommandSeq(commandseq, log, cwd)

    (output, error, status) = runInShell(commandseq, log, cwd)
    output = logOutput(log, output,logOutputflg)
    checkCommand(commandseq, status, output, error)
    return (output, error, status)

  def loadConfigure(self, argDB = None):
    if argDB is None:
      argDB = self.argDB
    if not 'configureCache' in argDB:
      self.logPrint('No cached configure in RDict at '+str(argDB.saveFilename))
      return None
    try:
      cache = argDB['configureCache']
      framework = pickle.loads(cache)
      framework.framework = framework
      framework.argDB = argDB
      self.logPrint('Loaded configure to cache: size '+str(len(cache)))
    except pickle.UnpicklingError as e:
      framework = None
      self.logPrint('Invalid cached configure: '+str(e))
    return framework

import args

class LanguageProcessor(args.ArgumentProcessor):
  def __init__(self, clArgs = None, argDB = None, framework = None, versionControl = None):
    self.languageModule      = {}
    self.preprocessorObject  = {}
    self.compilerObject      = {}
    self.linkerObject        = {}
    self.sharedLinkerObject  = {}
    self.dynamicLinkerObject = {}
    self.framework           = framework
    self.versionControl      = versionControl
    args.ArgumentProcessor.__init__(self, clArgs, argDB)
    self.outputFiles         = {}
    self.modulePath          = 'config.compile'
    return

  def getCompilers(self):
    if self.framework is None:
      return
    return self.framework.require('config.compilers', None)
  compilers = property(getCompilers, doc = 'The config.compilers configure object')
  def getLibraries(self):
    if self.framework is None:
      return
    return self.framework.require('config.libraries', None)
  libraries = property(getLibraries, doc = 'The config.libraries configure object')

  def __getstate__(self, d = None):
    '''We only want to pickle the language module names and output files. The other objects are set by configure.'''
    if d is None:
      d = args.ArgumentProcessor.__getstate__(self)
    if 'languageModule' in d:
      d['languageModule'] = dict([(lang,mod._loadName) for lang,mod in d['languageModule'].items()])
    for member in ['preprocessorObject', 'compilerObject', 'linkerObject', 'sharedLinkerObject', 'dynamicLinkerObject', 'framework']:
      if member in d:
        del d[member]
    return d

  def __setstate__(self, d):
    '''We must create the language modules'''
    args.ArgumentProcessor.__setstate__(self, d)
    self.__dict__.update(d)
    [self.getLanguageModule(language, moduleName) for language,moduleName in self.languageModule.items()]
    self.preprocessorObject  = {}
    self.compilerObject      = {}
    self.linkerObject        = {}
    self.sharedLinkerObject  = {}
    self.dynamicLinkerObject = {}
    return

  def setArgDB(self, argDB):
    args.ArgumentProcessor.setArgDB(self, argDB)
    for obj in self.preprocessorObject.values():
      if not hasattr(obj, 'argDB') or not obj.argDB == argDB:
        obj.argDB = argDB
    for obj in self.compilerObject.values():
      if not hasattr(obj, 'argDB') or not obj.argDB == argDB:
        obj.argDB = argDB
    for obj in self.linkerObject.values():
      if not hasattr(obj, 'argDB') or not obj.argDB == argDB:
        obj.argDB = argDB
    for obj in self.sharedLinkerObject.values():
      if not hasattr(obj, 'argDB') or not obj.argDB == argDB:
        obj.argDB = argDB
    for obj in self.dynamicLinkerObject.values():
      if not hasattr(obj, 'argDB') or not obj.argDB == argDB:
        obj.argDB = argDB
    if not self.compilers is None:
      self.compilers.argDB = argDB
      for obj in self.preprocessorObject.values():
        if hasattr(obj, 'configCompilers'):
          obj.configCompilers.argDB = argDB
      for obj in self.compilerObject.values():
        if hasattr(obj, 'configCompilers'):
          obj.configCompilers.argDB = argDB
      for obj in self.linkerObject.values():
        if hasattr(obj, 'configCompilers'):
          obj.configCompilers.argDB = argDB
      for obj in self.sharedLinkerObject.values():
        if hasattr(obj, 'configCompilers'):
          obj.configCompilers.argDB = argDB
      for obj in self.dynamicLinkerObject.values():
        if hasattr(obj, 'configCompilers'):
          obj.configCompilers.argDB = argDB
    if not self.libraries is None:
      self.libraries.argDB = argDB
      for obj in self.linkerObject.values():
        if hasattr(obj, 'configLibraries'):
          obj.configLibraries.argDB = argDB
      for obj in self.sharedLinkerObject.values():
        if hasattr(obj, 'configLibraries'):
          obj.configLibraries.argDB = argDB
      for obj in self.dynamicLinkerObject.values():
        if hasattr(obj, 'configLibraries'):
          obj.configLibraries.argDB = argDB
    return
  argDB = property(args.ArgumentProcessor.getArgDB, setArgDB, doc = 'The RDict argument database')

  def getLanguageModule(self, language, moduleName = None):
    '''Return the module associated with operations for a given language
       - Giving a moduleName explicitly forces a reimport'''
    if not language in self.languageModule or not moduleName is None:
      try:
        if moduleName is None:
          moduleName = self.modulePath+'.'+language
        module     = __import__(moduleName)
      except ImportError as e:
        if not moduleName is None:
          self.logPrint('Failure to find language module: '+str(e))
        try:
          moduleName = self.modulePath+'.'+language
          module     = __import__(moduleName)
        except ImportError as e:
          self.logPrint('Failure to find language module: '+str(e))
          moduleName = 'config.compile.'+language
          module     = __import__(moduleName)
      components = moduleName.split('.')
      for component in components[1:]:
        module   = getattr(module, component)
      module._loadName = moduleName
      self.languageModule[language] = module
    return self.languageModule[language]

  def getPreprocessorObject(self, language):
    if not language in self.preprocessorObject:
      self.preprocessorObject[language] = self.getLanguageModule(language).Preprocessor(self.argDB)
      self.preprocessorObject[language].setup()
    if not self.compilers is None:
      self.preprocessorObject[language].configCompilers = self.compilers
    if not self.versionControl is None:
      self.preprocessorObject[language].versionControl  = self.versionControl
    return self.preprocessorObject[language]

  def setPreprocessorObject(self, language, preprocessor):
    self.preprocessorObject[language] = preprocessor
    return self.getPreprocessorObject(language)

  def getCompilerObject(self, language):
    if not language in self.compilerObject:
      self.compilerObject[language] = self.getLanguageModule(language).Compiler(self.argDB)
      self.compilerObject[language].setup()
    if not self.compilers is None:
      self.compilerObject[language].configCompilers = self.compilers
    if not self.versionControl is None:
      self.compilerObject[language].versionControl  = self.versionControl
    return self.compilerObject[language]

  def setCompilerObject(self, language, compiler):
    self.compilerObject[language] = compiler
    return self.getCompilerObject(language)

  def getLinkerObject(self, language):
    if not language in self.linkerObject:
      self.linkerObject[language] = self.getLanguageModule(language).Linker(self.argDB)
      self.linkerObject[language].setup()
    if not self.compilers is None:
      self.linkerObject[language].configCompilers = self.compilers
    if not self.libraries is None:
      self.linkerObject[language].configLibraries = self.libraries
    if not self.versionControl is None:
      self.linkerObject[language].versionControl  = self.versionControl
    return self.linkerObject[language]

  def setLinkerObject(self, language, linker):
    self.linkerObject[language] = linker
    return self.getLinkerObject(language)

  def getSharedLinkerObject(self, language):
    if not language in self.sharedLinkerObject:
      self.sharedLinkerObject[language] = self.getLanguageModule(language).SharedLinker(self.argDB)
      self.sharedLinkerObject[language].setup()
    if not self.compilers is None:
      self.sharedLinkerObject[language].configCompilers = self.compilers
    if not self.libraries is None:
      self.sharedLinkerObject[language].configLibraries = self.libraries
    if not self.versionControl is None:
      self.sharedLinkerObject[language].versionControl  = self.versionControl
    return self.sharedLinkerObject[language]

  def setSharedLinkerObject(self, language, linker):
    self.sharedLinkerObject[language] = linker
    return self.getSharedLinkerObject(language)

  def getDynamicLinkerObject(self, language):
    if not language in self.dynamicLinkerObject:
      self.dynamicLinkerObject[language] = self.getLanguageModule(language).DynamicLinker(self.argDB)
      self.dynamicLinkerObject[language].setup()
    if not self.compilers is None:
      self.dynamicLinkerObject[language].configCompilers = self.compilers
    if not self.libraries is None:
      self.dynamicLinkerObject[language].configLibraries = self.libraries
    if not self.versionControl is None:
      self.dynamicLinkerObject[language].versionControl  = self.versionControl
    return self.dynamicLinkerObject[language]

  def setDynamicLinkerObject(self, language, linker):
    self.dynamicLinkerObject[language] = linker
    return self.getDynamicLinkerObject(language)
