import sys
if not hasattr(sys, 'version_info'):
  print '*** Python version 1 is not supported. Please get the latest version from www.python.org ***'
  sys.exit(4)

import nargs
useThreads = nargs.Arg.findArgument('useThreads', sys.argv[1:])
if useThreads is None:
  useThreads = 1
else:
  useThreads = int(useThreads)

import logging

class Script(logging.Logger):
  def __init__(self, clArgs = None, argDB = None):
    self.checkPython()
    logging.Logger.__init__(self, clArgs, argDB)
    self.shell = '/bin/sh'
    self.getRoot()
    return

  def setupArguments(self, argDB):
    '''This method now also creates the help and action logs'''
    import help

    argDB = logging.Logger.setupArguments(self, argDB)

    self.help = help.Help(argDB)
    self.help.title = 'Script Help'

    self.actions = help.Info()
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
    logging.Logger.setup(self)
    if self.argDB['help'] or self.argDB['h']:
      import sys

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

  def getRoot(self):
    '''Return the directory containing this module
       - This has the problem that when we reload a module of the same name, this gets screwed up
         Therefore, we call it in the initializer, and stash it'''
    if not hasattr(self, '_root_'):
      import os
      import sys

      # Work around a bug with pdb in 2.3
      if hasattr(sys.modules[self.__module__], '__file__') and not os.path.basename(sys.modules[self.__module__].__file__) == 'pdb.py':
        self._root_ = os.path.abspath(os.path.dirname(sys.modules[self.__module__].__file__))
      else:
        self._root_ = os.getcwd()
    return self._root_

  def getModule(self, root, name):
    '''Retrieve a specific module from the directory root, bypassing the usual paths'''
    import imp

    (fp, pathname, description) = imp.find_module(name, [root])
    try:
      return imp.load_module(name, fp, pathname, description)
    finally:
      if fp: fp.close()
    return

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
    while 1:
      ready = select.select([output, error], [], [])
      if len(ready[0]):
        if error in ready[0]:
          msg = error.readline()
          if msg:
            err += msg
          else:
            errorClosed = 1
        if output in ready[0]:
          msg = output.readline()
          if msg:
            out += msg
          else:
            outputClosed = 1
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

  def executeShellCommand(command, checkCommand = None, timeout = 120.0, log = None):
    '''Execute a shell command returning the output, and optionally provide a custom error checker'''
    import threading
    global output, error, status

    def logOutput(log, output):
      import re
      # get rid of multiple blank lines
      output = re.sub('\n[\n]*','\n', output)
      if len(output) < 600:
        if log: log.write('sh: '+output+'\n')
      else:
        if log:
          log.write('sh: '+output[:600]+'...\n')
          log.write('... '+output[-600:]+'\n')
      return output

    if log is None:
      log = logging.Logger.defaultLog
    if log:
      log.write('sh: '+command+'\n')
    if useThreads:
      status = -1
      output = 'Runaway process'
      def run(command, log):
        global output, error, status
        (output, error, status) = Script.runShellCommand(command, log)
        return

      thread = threading.Thread(target = run, name = 'Shell Command', args = (command, log))
      thread.setDaemon(1)
      thread.start()
      thread.join(timeout)
      if thread.isAlive():
        error  = 'Runaway process exceeded time limit of '+str(timeout)+'s\n'
        status = -1
        if log: log.write(error)
      else:
        output = logOutput(log, output)
    else:
      (output, error, status) = Script.runShellCommand(command, log)
      output                  = logOutput(log, output)
    if checkCommand:
      checkCommand(command, status, output, error)
    else:
      Script.defaultCheckCommand(command, status, output, error)
    return (output, error, status)
  executeShellCommand = staticmethod(executeShellCommand)

import args

class LanguageProcessor(args.ArgumentProcessor):
  def __init__(self, clArgs = None, argDB = None):
    args.ArgumentProcessor.__init__(self, clArgs, argDB)
    self.languageModule     = {}
    self.preprocessorObject = {}
    self.compilerObject     = {}
    self.linkerObject       = {}
    self.modulePath         = 'config.compile'
    return

  def __getstate__(self, d = None):
    '''We do not want to pickle the language modules'''
    if d is None:
      d = args.ArgumentProcessor.__getstate__(self)
    if 'languageModule' in d:
      d['languageModule'] = dict([(lang,mod._loadName) for lang,mod in d['languageModule'].items()])
    return d

  def __setstate__(self, d):
    '''We must create the language modules'''
    args.ArgumentProcessor.__setstate__(self, d)
    self.__dict__.update(d)
    [self.getLanguageModule(language, moduleName) for language,moduleName in self.languageModule.items()]
    return

  def getLanguageModule(self, language, moduleName = None):
    '''Return the module associated with operations for a given language
       - Giving a moduleName explicitly forces a reimport'''
    if not language in self.languageModule or not moduleName is None:
      try:
        if moduleName is None:
          moduleName = self.modulePath+'.'+language.replace('+', 'x')
        module     = __import__(moduleName)
      except ImportError, e:
        if not moduleName is None:
          self.logPrint('Failure to find language module: '+str(e))
        try:
          moduleName = self.modulePath+'.'+language.replace('+', 'x')
          module     = __import__(moduleName)
        except ImportError, e:
          self.logPrint('Failure to find language module: '+str(e))
          moduleName = 'config.compile.'+language.replace('+', 'x')
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
    return self.preprocessorObject[language]

  def getCompilerObject(self, language):
    if not language in self.compilerObject:
      self.compilerObject[language] = self.getLanguageModule(language).Compiler(self.argDB)
    return self.compilerObject[language]

  def getLinkerObject(self, language):
    if not language in self.linkerObject:
      self.linkerObject[language] = self.getLanguageModule(language).Linker(self.argDB)
    return self.linkerObject[language]
