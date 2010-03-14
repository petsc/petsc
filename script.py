import sys
if not hasattr(sys, 'version_info'):
  print '*** Python version 1 is not supported. Please get the latest version from www.python.org ***'
  sys.exit(4)

import cPickle

try:
  import subprocess
  USE_SUBPROCESS = 1
except ImportError:
  USE_SUBPROCESS = 0

# Some features related to detecting login failures cannot be easily
# implemented with the 'subprocess' module. Disable it for now ...
USE_SUBPROCESS = 0
# In Python 2.6 and above, the 'popen2' module is deprecated
if sys.version_info[:2] >= (2, 6) and not USE_SUBPROCESS:
  import warnings
  warnings.filterwarnings('ignore', category=DeprecationWarning, module=__name__)

import nargs
useThreads = nargs.Arg.findArgument('useThreads', sys.argv[1:])
if useThreads is None:
  useThreads = 1
else:
  useThreads = int(useThreads)

import logger

class Script(logger.Logger):
  def __init__(self, clArgs = None, argDB = None):
    self.checkPython()
    logger.Logger.__init__(self, clArgs, argDB)
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

    help.addArgument('Script', '-help', nargs.ArgBool(None, 0, 'Print this help message', isTemporary = 1), ignoreDuplicates = 1)
    help.addArgument('Script', '-h',    nargs.ArgBool(None, 0, 'Print this help message', isTemporary = 1), ignoreDuplicates = 1)
    return help

  def setup(self):
    ''' This method checks to see whether help was requested'''
    if hasattr(self, '_setup'):
      return
    logger.Logger.setup(self)
    self._setup = 1
    if self.hasHelpFlag():
      self.help.output()
      sys.exit()
    return

  def cleanup(self):
    '''This method outputs the action log'''
    self.actions.output(self.log)
    return

  def checkPython(self):
    if not hasattr(sys, 'version_info') or float(sys.version_info[0]) < 2 or float(sys.version_info[1]) < 2:
      raise RuntimeError('BuildSystem requires Python version 2.2 or higher. Get Python at http://www.python.org')
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

  if USE_SUBPROCESS:

    def runShellCommand(command, log=None):
      Popen = subprocess.Popen
      PIPE  = subprocess.PIPE
      if log: log.write('Executing: '+command+'\n')
      pipe = Popen(command, stdin=None, stdout=PIPE, stderr=PIPE,
                   bufsize=-1, shell=True, universal_newlines=True)
      (out, err) = pipe.communicate()
      ret = pipe.returncode
      return (out, err, ret)

  else:

    def openPipe(command):
      '''We need to use the asynchronous version here since we want to avoid blocking reads'''
      import popen2

      pipe = None
      if hasattr(popen2, 'Popen3'):
        pipe   = popen2.Popen3(command, 1)
        input  = pipe.tochild
        output = pipe.fromchild
        err    = pipe.childerr
      else:
        import os
        (input, output, err) = os.popen3(command)
      return (input, output, err, pipe)
    openPipe = staticmethod(openPipe)

    def runShellCommand(command, log = None):
      import select

      ret        = None
      out        = ''
      err        = ''
      loginError = 0
      if log: log.write('Executing: '+command+'\n')
      (input, output, error, pipe) = Script.openPipe(command)
      input.close()
      outputClosed = 0
      errorClosed  = 0
      lst = [output, error]
      while 1:
        ready = select.select(lst, [], [])
        if len(ready[0]):
          if error in ready[0]:
            msg = error.readline()
            if msg:
              err += msg
            else:
              errorClosed = 1
              lst.remove(error)
          if output in ready[0]:
            msg = output.readline()
            if msg:
              out += msg
            else:
              outputClosed = 1
              lst.remove(output)
          if out.find('password:') >= 0 or err.find('password:') >= 0:
            loginError = 1
            break
        if outputClosed and errorClosed:
          break
      output.close()
      error.close()
      if pipe:
        # We would like the NOHANG argument here
        ret = pipe.wait()
      if loginError:
        raise RuntimeError('Could not login to site')
      return (out, err, ret)

  runShellCommand = staticmethod(runShellCommand)

  def defaultCheckCommand(command, status, output, error):
    '''Raise an error if the exit status is nonzero'''
    if status: raise RuntimeError('Could not execute \''+command+'\':\n'+output+error)
  defaultCheckCommand = staticmethod(defaultCheckCommand)

  def executeShellCommand(command, checkCommand = None, timeout = 600.0, log = None):
    '''Execute a shell command returning the output, and optionally provide a custom error checker
       - This returns a tuple of the (output, error, statuscode)'''
    if not checkCommand:
      checkCommand = Script.defaultCheckCommand
    if log is None:
      log = logger.Logger.defaultLog
    def logOutput(log, output):
      import re
      # get rid of multiple blank lines
      output = re.sub('\n[\n]*','\n', output)
      log.write('sh: '+output+'\n')
      return output
    def runInShell(command, log):
      if useThreads:
        import threading
        class InShell(threading.Thread):
          def __init__(self):
            threading.Thread.__init__(self)
            self.name = 'Shell Command'
            self.setDaemon(1)
          def run(self):
            (self.output, self.error, self.status) = Script.runShellCommand(command, log)
        thread = InShell()
        thread.start()
        thread.join(timeout)
        if thread.isAlive():
          error = 'Runaway process exceeded time limit of '+str(timeout)+'s\n'
          log.write(error)
          return ('', error, -1)
        else:
          return (thread.output, thread.error, thread.status)
      else:
        return Script.runShellCommand(command, log)

    log.write('sh: '+command+'\n')
    (output, error, status) = runInShell(command, log)
    output = logOutput(log, output)
    checkCommand(command, status, output, error)
    return (output, error, status)
  executeShellCommand = staticmethod(executeShellCommand)

  def getDebugger(self, className = 'PETSc.DebugI.GDB.Debugger'):
    if not hasattr(self, '_debugger'):
      try:
        import SIDL.Loader
      except ImportError, e:
        self.logPrint('Cannot locate a functional SIDL loader: '+str(e))
        return
      try:
        import PETSc.Debug.Debugger
      except ImportError, e:
        self.logPrint('Could not load Petsc debugger module: '+str(e))
        return
      debugger = PETSc.Debug.Debugger.Debugger(SIDL.Loader.createClass(className))
      if not debugger:
        self.logPrint('Could not load debugger: '+cls)
        return
      debugger.setProgram('/usr/local/python/bin/python')
      debugger.setUseXterm(1)
      debugger.setDebugger('/usr/local/gdb/bin/gdb')
      debugger.attach()
      self._debugger = debugger
    return self._debugger
  def setDebugger(self, debugger):
    self._debugger = debugger
    return
  debugger = property(getDebugger, setDebugger, doc = 'The debugger')

  def loadConfigure(self, argDB = None):
    if argDB is None:
      argDB = self.argDB
    if not 'configureCache' in argDB:
      self.logPrint('No cached configure in RDict')
      return None
    try:
      cache = argDB['configureCache']
      framework = cPickle.loads(cache)
      framework.framework = framework
      framework.argDB = argDB
      self.logPrint('Loaded configure to cache: size '+str(len(cache)))
    except cPickle.UnpicklingError, e:
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
      except ImportError, e:
        if not moduleName is None:
          self.logPrint('Failure to find language module: '+str(e))
        try:
          moduleName = self.modulePath+'.'+language
          module     = __import__(moduleName)
        except ImportError, e:
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
