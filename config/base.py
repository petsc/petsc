import os
import re
import select
import sys

#Determine whether we can safely use threads
import nargs
useThreads = nargs.Arg.findArgument('useThreads', sys.argv[1:])
if useThreads is None:
  useThreads = 1
else:
  useThreads = int(useThreads)

class Configure:
  def __init__(self, framework):
    self.framework       = framework
    self.defines         = {}
    self.subst           = {}
    self.argSubst        = {}
    self.shell           = '/bin/sh'
    self.language        = []
    self.compilerDefines = 'confdefs.h'
    self.pushLanguage('C')
    return

  def __str__(self):
    return ''

  def checkPython(self):
    if not hasattr(sys, 'version_info') or float(sys.version_info[0]) < 2 or float(sys.version_info[1]) < 2:
      raise RuntimeError('BuildSystem requires Python version 2.2 or higher. Get Python at http://www.python.org')
    return

  def getRoot(self):
    # This has the problem that when we reload a module of the same name, this gets screwed up
    #   Therefore, we call it in the initializer, and stash it
    if not hasattr(self, '_root_'):
      if hasattr(sys.modules[self.__module__], '__file__'):
        self._root_ = os.path.abspath(os.path.dirname(sys.modules[self.__module__].__file__))
      else:
        self._root_ = os.getcwd()
    return self._root_

  def startLine(self):
    '''Erases last print line and puts cursor at first point in line'''
    if self.framework.linewidth < 0: return
    self.printLine('')
    for i in range(0,self.framework.linewidth):
      sys.stdout.write('\b')

  def printLine(self,msg):
    if not hasattr(self.framework,'linewidth'):
      self.framework.cwd       = os.getcwd()+'/'
      if 'with-scroll-output' in self.framework.argDB and self.framework.argDB['with-scroll-output']:
        self.framework.linewidth = -2
      elif 'with-no-output' in self.framework.argDB and self.framework.argDB['with-no-output']:
        self.framework.linewidth = -1
      else:
        if not sys.stdout.isatty():
          self.framework.linewidth = -2            
        try:
          import curses
          try:
            curses.setupterm()
            stdscr = curses.initscr()
            (y,x) = stdscr.getmaxyx()
            curses.endwin()

            self.framework.linewidth = x
          except curses.error:
            self.framework.linewidth = -2            
        except ImportError:
          self.framework.linewidth = -2
            
    if self.framework.linewidth == -1:
      return
    elif self.framework.linewidth > 0:
      for i in range(0,self.framework.linewidth):
        sys.stdout.write('\b')
      msg = msg.replace(self.framework.cwd,'')
      msg = msg+'                                                                                                                       '
      sys.stdout.write(msg[0:self.framework.linewidth])
      sys.stdout.flush()
    else:
      msg = msg.replace(self.framework.cwd,'')
      print msg      
    return

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

    if log: log.write('sh: '+command+'\n')
    if useThreads:
      status = -1
      output = 'Runaway process'
      def run(command, log):
        global output, error, status
        (output, error, status) = Configure.runShellCommand(command, log)
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
      (output, error, status) = Configure.runShellCommand(command, log)
      output                  = logOutput(log, output)
    if checkCommand:
      checkCommand(command, status, output, error)
    else:
      Configure.defaultCheckCommand(command, status, output, error)
    return (output, error, status)
  executeShellCommand = staticmethod(executeShellCommand)

  def executeTest(self, test, args = []):
    self.framework.log.write('================================================================================\n')
    self.framework.log.write('TEST '+str(test.im_func.func_name)+' from '+str(test.im_class.__module__)+'('+str(test.im_func.func_code.co_filename)+':'+str(test.im_func.func_code.co_firstlineno)+')\n')
    self.printLine('TESTING: '+str(test.im_func.func_name)+' from '+str(test.im_class.__module__)+'('+str(test.im_func.func_code.co_filename)+':'+str(test.im_func.func_code.co_firstlineno)+')')
    if test.__doc__: self.framework.log.write('  '+test.__doc__+'\n')
    if not isinstance(args, list): args = [args]
    return apply(test, args)

  #################################
  # Define and Substitution Support
  def addDefine(self, name, value):
    '''Designate that "name" should be defined to "value" in the configuration header'''
    self.framework.log.write('Defined '+name+' to '+str(value)+' in '+str(self.__module__)+'\n')
    self.defines[name] = value
    return

  def addSubstitution(self, name, value):
    '''Designate that "@name@" should be replaced by "value" in all files which experience substitution'''
    self.framework.log.write('Substituting '+name+' with '+str(value)+' in '+str(self.__module__)+'\n')
    self.subst[name] = value
    return

  def addArgumentSubstitution(self, name, arg):
    '''Designate that "@name@" should be replaced by "arg" in all files which experience substitution'''
    self.framework.log.write('Substituting '+name+' with '+str(arg)+'('+str(self.framework.argDB[arg])+') in '+str(self.__module__)+'\n')
    self.argSubst[name] = arg
    return

  ################
  # Program Checks
  def checkExecutable(self, dir, name):
    prog  = os.path.join(dir, name)
    found = 0
    self.framework.log.write('Checking for program '+prog+'...')
    if os.path.isfile(prog) and os.access(prog, os.X_OK):
      found = 1
      self.framework.log.write('found\n')
    else:
      self.framework.log.write('not found\n')
    return found

  def getExecutable(self, name, path = [], getFullPath = 0, useDefaultPath = 0, resultName = ''):
    found = 0
    index = name.find(' ')
    if index >= 0:
      options = name[index:]
      name    = name[:index]
    else:
      options = ''
    if not resultName:
      resultName = name
    if isinstance(path, str):
      path = path.split(':')
    if not len(path):
      useDefaultPath = 1
    for dir in path:
      if self.checkExecutable(dir, name):
        found       = 1
        getFullPath = 1
        break
    if useDefaultPath and not found:
      for dir in os.environ['PATH'].split(':'):
        if self.checkExecutable(dir, name):
          found     = 1
          break
    if not found:
      for dir in self.framework.argDB['search-dirs']:
        if self.checkExecutable(dir, name):
          found       = 1
          getFullPath = 1
          break
    if found:
      if getFullPath:
        setattr(self, resultName, os.path.abspath(os.path.join(dir, name))+options)
      else:
        setattr(self, resultName, name+options)
      self.addSubstitution(resultName.upper(), getattr(self, resultName))
    return found

  def getExecutables(self, names, path = '', getFullPath = 0, resultName = ''):
    for name in names:
      if self.getExecutable(name, path = path, getFullPath = getFullPath, useDefaultPath = 0, resultName = resultName):
        return name
    return None

  ###############################################
  # Preprocessor, Compiler, and Linker Operations
  def pushLanguage(self, language):
    self.language.append(language)
    return self.language[-1]

  def popLanguage(self):
    self.language.pop()
    return self.language[-1]

  def getCompiler(self):
    compiler            = self.framework.getCompilerObject(self.language[-1])
    compiler.checkSetup()
    self.compilerSource = 'conftest'+compiler.sourceExtension
    self.compilerObj    = 'conftest'+compiler.targetExtension
    return self.framework.argDB[compiler.name]

  def getCompilerFlags(self):
    return self.framework.getCompilerObject(self.language[-1]).getFlags()

  def getLinker(self):
    linker            = self.framework.getLinkerObject(self.language[-1])
    linker.checkSetup()
    self.linkerSource = 'conftest'+linker.sourceExtension
    self.linkerObj    = 'conftest'
    return self.framework.argDB[linker.name]

  def getLinkerFlags(self):
    return self.framework.getLinkerObject(self.language[-1]).getFlags()

  def getPreprocessorCmd(self):
    self.getCompiler()
    preprocessor = self.framework.getPreprocessorObject(self.language[-1])
    preprocessor.checkSetup()
    return preprocessor.getCommand(self.compilerSource)

  def getCompilerCmd(self):
    self.getCompiler()
    compiler = self.framework.getCompilerObject(self.language[-1])
    compiler.checkSetup()
    return compiler.getCommand(self.compilerSource, self.compilerObj)

  def getLinkerCmd(self):
    self.getLinker()
    linker = self.framework.getLinkerObject(self.language[-1])
    linker.checkSetup()
    return linker.getCommand(self.linkerSource, self.linkerObj)

  def getCode(self, includes, body = None, codeBegin = None, codeEnd = None):
    language = self.language[-1]
    if includes and not includes[-1] == '\n':
      includes += '\n'
    if language in ['C', 'C++', 'Cxx']:
      codeStr = '#include "confdefs.h"\n'+includes
      if not body is None:
        if codeBegin is None:
          codeBegin = '\nint main() {\n'
        if codeEnd is None:
          codeEnd   = ';\n  return 0;\n}\n'
        codeStr += codeBegin+body+codeEnd
    elif language == 'F77':
      if not includes is None:
        codeStr = includes
      else:
        codeStr = ''
      if not body is None:
        if codeBegin is None:
          codeBegin = '      program main\n'
        if codeEnd is None:
          codeEnd   = '\n      end\n'
        codeStr += codeBegin+body+codeEnd
    else:
      raise RuntimeError('Invalid language: '+language)
    return codeStr

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
    ret     = None
    out     = ''
    err     = ''
    if log: log.write('Executing: '+command+'\n')
    (input, output, error, pipe) = Configure.openPipe(command)
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
      if outputClosed and errorClosed:
        break
    output.close()
    error.close()
    if pipe:
      # We would like the NOHANG argument here
      ret = pipe.wait()
    return (out, err, ret)
  runShellCommand = staticmethod(runShellCommand)

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
    (out, err, ret) = Configure.executeShellCommand(command, checkCommand = report, log = self.framework.log)
    if os.path.isfile(self.compilerDefines): os.remove(self.compilerDefines)
    if os.path.isfile(self.compilerSource): os.remove(self.compilerSource)
    return (out, err, ret)

  def outputPreprocess(self, codeStr):
    '''Return the contents of stdout when preprocessing "codeStr"'''
    return self.preprocess(codeStr)[0]

  def checkPreprocess(self, codeStr):
    '''Return True if an error occurred
       - An error is signaled by a nonzero return code, or output on stderr'''
    (out, err, ret) = self.preprocess(codeStr)
    return not ret and not len(err)

  def filterCompileOutput(self, output):
    return self.framework.filterCompileOutput(output)

  def outputCompile(self, includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None):
    '''Return the error output from this compile and the return code'''
    def report(command, status, output, error):
      if error or status:
        self.framework.log.write('Possible ERROR while running compiler: '+output)
        if status: self.framework.log.write('ret = '+str(status)+'\n')
        if error: self.framework.log.write('error message = {'+error+'}\n')
        self.framework.log.write('Source:\n'+self.getCode(includes, body, codeBegin, codeEnd))
      return

    command = self.getCompilerCmd()
    self.framework.outputHeader(self.compilerDefines)
    f = file(self.compilerSource, 'w')
    f.write(self.getCode(includes, body, codeBegin, codeEnd))
    f.close()
    (out, err, ret) = Configure.executeShellCommand(command, checkCommand = report, log = self.framework.log)
    if os.path.isfile(self.compilerDefines): os.remove(self.compilerDefines)
    if os.path.isfile(self.compilerSource): os.remove(self.compilerSource)
    if cleanup and os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    return (out, err, ret)

  def checkCompile(self, includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None):
    '''Returns True if the compile was successful'''
    (output, error, returnCode) = self.outputCompile(includes, body, cleanup)
    output = self.filterCompileOutput(output+'\n'+error)
    return not (returnCode or len(output))

  def getCompilerFlagsArg(self, compilerOnly = 0):
    '''Return the name of the argument which holds the compiler flags for the current language'''
    language = self.language[-1]
    if language == 'C':
      flagsArg = 'CFLAGS'
    elif language in ['C++', 'Cxx']:
      if compilerOnly:
        flagsArg = 'CXX_CXXFLAGS'
      else:
        flagsArg = 'CXXFLAGS'
    elif language == 'F77':
      flagsArg = 'FFLAGS'
    else:
      raise RuntimeError('Unknown language: '+language)
    return flagsArg

  def checkCompilerFlag(self, flag, includes = '', body = '', compilerOnly = 0):
    '''Determine whether the compiler accepts the given flag'''
    flagsArg = self.getCompilerFlagsArg(compilerOnly)
    oldFlags = self.framework.argDB[flagsArg]
    self.framework.argDB[flagsArg] = self.framework.argDB[flagsArg]+' '+flag
    (output, error, status)        = self.outputCompile(includes, body)
    output  += error
    valid    = 1
    if status or output.find('unrecognized option') >= 0 or output.find('unknown flag') >= 0 or output.find('unknown option') >= 0 or output.find('ignoring option') >= 0 or output.find('not recognized') >= 0 or output.find('ignored') >= 0 or output.find('illegal option') >= 0  or output.find('linker input file unused because linking not done') >= 0 or output.find('Unknown switch') >= 0:
      valid = 0
    self.framework.argDB[flagsArg] = oldFlags
    return valid

  def addCompilerFlag(self, flag, includes = '', body = '', extraflags = '', compilerOnly = 0):
    '''Determine whether the compiler accepts the given flag, and add it if valid'''
    if self.checkCompilerFlag(flag+' '+extraflags, includes, body, compilerOnly):
      flagsArg = self.getCompilerFlagsArg(compilerOnly)
      self.framework.argDB[flagsArg] = self.framework.argDB[flagsArg]+' '+flag
      self.framework.log.write('Added '+self.language[-1]+' compiler flag '+flag+'\n')
      return
    raise RuntimeError('Bad compiler flag: '+flag)

  def filterLinkOutput(self, output):
    return self.framework.filterLinkOutput(output)

  def outputLink(self, includes, body, cleanup = 1, codeBegin = None, codeEnd = None):
    (out, err, ret) = self.outputCompile(includes, body, cleanup = 0, codeBegin = codeBegin, codeEnd = codeEnd)
    out = self.filterCompileOutput(out+'\n'+err)
    if ret or len(out):
      return (out, ret)

    def report(command, status, output, error):
      if error or status:
        self.framework.log.write('Possible ERROR while running linker: '+error)
        self.framework.log.write(' output: '+output)
        if status: self.framework.log.write('ret = '+str(status)+'\n')
        if error: self.framework.log.write('error message = {'+error+'}\n')
        self.framework.log.write(' in '+self.getLinkerCmd()+'\n')
        self.framework.log.write('Source:\n'+self.getCode(includes, body, codeBegin, codeEnd))
      return

    (out, err, ret) = Configure.executeShellCommand(self.getLinkerCmd(), checkCommand = report, log = self.framework.log)
    if sys.platform[:3] == 'win' or sys.platform == 'cygwin':
      self.linkerObj = self.linkerObj+'.exe'
    if os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    if cleanup and os.path.isfile(self.linkerObj): os.remove(self.linkerObj)
    return (out+err, ret)

  def checkLink(self, includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None):
    (output, returnCode) = self.outputLink(includes, body, cleanup, codeBegin, codeEnd)
    output = self.filterLinkOutput(output)
    return not (returnCode or len(output))

  def getLinkerFlagsArg(self):
    '''Return the name of the argument which holds the linker flags for the current language'''
    language = self.language[-1]
    if language == 'C':
      flagsArg = 'LDFLAGS'
    elif language in ['C++', 'Cxx']:
      flagsArg = 'LDFLAGS'
    elif language == 'F77':
      flagsArg = 'LDFLAGS'
    else:
      raise RuntimeError('Unknown language: '+language)
    return flagsArg

  def checkLinkerFlag(self, flag):
    '''Determine whether the linker accepts the given flag'''
    flagsArg = self.getLinkerFlagsArg()
    oldFlags = self.framework.argDB[flagsArg]
    valid    = 1
    self.framework.argDB[flagsArg] = self.framework.argDB[flagsArg]+' '+flag
    (output, status)               = self.outputLink('', '')
    if status:
      valid = 0
      self.framework.log.write('Rejecting linker flag '+flag+' due to nonzero status from link\n')
    if output.find('unrecognized option') >= 0 or output.find('unknown flag') >= 0 or (output.find('bad ') >= 0 and output.find(' option') >= 0) or output.find('linker input file unused because linking not done') >= 0 or output.find('flag is ignored') >= 0 or output.find('Invalid option') >= 0 or output.find('unknown option') >= 0:
      valid = 0
      self.framework.log.write('Rejecting linker flag '+flag+' due to \n'+output)
    self.framework.argDB[flagsArg] = oldFlags
    return valid

  def addLinkerFlag(self, flag):
    '''Determine whether the linker accepts the given flag, and add it if valid'''
    if self.checkLinkerFlag(flag):
      flagsArg = self.getLinkerFlagsArg()
      self.framework.argDB[flagsArg] = self.framework.argDB[flagsArg]+' '+flag
      return
    raise RuntimeError('Bad linker flag: '+flag)

  def outputRun(self, includes, body, cleanup = 1, defaultOutputArg = ''):
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
      (output, error, status) = Configure.executeShellCommand(command, log = self.framework.log)
    except RuntimeError, e:
      self.framework.log.write('ERROR while running executable: '+str(e)+'\n')
    if os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    if cleanup and os.path.isfile(self.linkerObj): os.remove(self.linkerObj)
    return (output+error, status)

  def checkRun(self, includes = '', body = '', cleanup = 1, defaultArg = ''):
    (output, returnCode) = self.outputRun(includes, body, cleanup, defaultArg)
    return not returnCode

  def splitLibs(self,libArgs):
    '''Takes a string containing a list of libraries (including potentially -L, -l, -w etc) and generates a list of libraries'''
    dirs = []
    libs = []
    for arg in libArgs.split(' '):
      if not arg: continue
      if arg.startswith('-L'):
        dirs.append(arg[2:])
      elif arg.startswith('-l'):
        libs.append(arg[2:])
      elif not arg.startswith('-'):
        libs.append(arg)
    libArgs = []
    for lib in libs:
      if not os.path.isabs(lib):
        added = 0
        for dir in dirs:
          if added:
	    break
	  for ext in ['a', 'so']:
            filename = os.path.join(dir, 'lib'+lib+'.'+ext)
            if os.path.isfile(filename):
              libArgs.append(filename)
              added = 1
              break
      else:
        libArgs.append(lib)
    return libArgs

  def splitIncludes(self,incArgs):
    '''Takes a string containing a list of include directories with -I and generates a list of includes'''
    includes = []
    for inc in incArgs.split(' '):
      if inc.startswith('-I'):
        # check if directory exists?
        includes.append(inc[2:])
    return includes

  def configure(self):
    pass
