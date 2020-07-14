'''
config.base.Configure is the base class for all configure objects. It handles several types of interaction:

Framework hooks
---------------

  The Framework will first instantiate the object and call setupDependencies(). All require()
  calls should be made in that method.

  The Framework will then call configure(). If it succeeds, the object will be marked as configured.

Generic test execution
----------------------

  All configure tests should be run using

  executeTest()

which formats the output and adds metadata for the log.

Preprocessing, Compiling, Linking, and Running
----------------------------------------------

  Two forms of this check are provided for each operation. The first is an "output" form which is
intended to provide the status and complete output of the command. The second, or "check" form will
return a success or failure indication based upon the status and output.

  outputPreprocess(), checkPreprocess(), preprocess()
  outputCompile(),    checkCompile()
  outputLink(),       checkLink()
  outputRun(),        checkRun()

  The language used for these operation is managed with a stack, similar to autoconf.

  pushLanguage(), popLanguage()

  We also provide special forms used to check for valid compiler and linker flags, optionally adding
them to the defaults.

  checkCompilerFlag(), addCompilerFlag()
  checkLinkerFlag(),   addLinkerFlag()

Finding Executables
-------------------

  getExecutable(), getExecutables(), checkExecutable()

Output
------

  addDefine(), addSubstitution(), addArgumentSubstitution(), addTypedef(), addPrototype()
  addMakeMacro(), addMakeRule()

  The object may define a headerPrefix member, which will be appended, followed
by an underscore, to every define which is output from it. Similarly, a substPrefix
can be defined which applies to every substitution from the object. Typedefs and
function prototypes are placed in a separate header in order to accomodate languges
such as Fortran whose preprocessor can sometimes fail at these statements.
'''
import script

import os
import time
import contextlib

class ConfigureSetupError(Exception):
  pass

class Configure(script.Script):
  def __init__(self, framework, tmpDir = None):
    script.Script.__init__(self, framework.clArgs, framework.argDB)
    self.framework       = framework
    self.defines         = {}
    self.makeRules       = {}
    self.makeMacros      = {}
    self.typedefs        = {}
    self.prototypes      = {}
    self.subst           = {}
    self.argSubst        = {}
    self.language        = []
    if not tmpDir is None:
      self.tmpDir        = tmpDir
    try:
      # The __init__ method may be called to reinitialize in the future (e.g.,
      # updateCompilers()) and will need to be re-setup in that case.
      delattr(self, '_setup')
    except AttributeError:
      pass
    return

  def setup(self):
    if hasattr(self, '_setup'):
      return
    script.Script.setup(self)
    self._setup = 1
    self.pushLanguage('C')

  def getTmpDir(self):
    if not hasattr(self, '_tmpDir'):
      self._tmpDir = os.path.join(self.framework.tmpDir, self.__module__)
      if not os.path.isdir(self._tmpDir): os.mkdir(self._tmpDir)
      self.logPrint('All intermediate test results are stored in '+self._tmpDir)
    return self._tmpDir
  def setTmpDir(self, temp):
    if hasattr(self, '_tmpDir'):
      if os.path.isdir(self._tmpDir):
        import shutil
        shutil.rmtree(self._tmpDir)
      if temp is None:
        delattr(self, '_tmpDir')
    if not temp is None:
      self._tmpDir = temp
    return
  tmpDir = property(getTmpDir, setTmpDir, doc = 'Temporary directory for test byproducts')

  def __str__(self):
    return ''

  def logError(self, component, status, output, error):
    if status:
      exitstr = ' exit code ' + str(status)
    else:
      exitstr = ''
    self.logWrite('Possible ERROR while running %s:%s\n' % (component, exitstr))
    if output:
      self.logWrite('stdout:\n' + output)
    if error:
      self.logWrite('stderr:\n' + error)

  def executeTest(self, test, args = [], kargs = {}):
    import time

    self.logWrite('================================================================================\n')
    self.logWrite('TEST '+str(test.__func__.__name__)+' from '+str(test.__self__.__class__.__module__)+'('+str(test.__func__.__code__.co_filename)+':'+str(test.__func__.__code__.co_firstlineno)+')\n')
    self.logPrint('TESTING: '+str(test.__func__.__name__)+' from '+str(test.__self__.__class__.__module__)+'('+str(test.__func__.__code__.co_filename)+':'+str(test.__func__.__code__.co_firstlineno)+')', debugSection = 'screen', indent = 0)
    if test.__doc__: self.logWrite('  '+test.__doc__+'\n')
    #t = time.time()
    if not isinstance(args, list): args = [args]
    ret = test(*args,**kargs)
    #self.logPrint('  TIME: '+str(time.time() - t)+' sec', debugSection = 'screen', indent = 0)
    return ret

  #################################
  # Define and Substitution Supported
  def addMakeRule(self, name, dependencies, rule = []):
    '''Designate that "name" should be rule in the makefile header (bmake file)'''
    self.logPrint('Defined make rule "'+name+'" with dependencies "'+str(dependencies)+'" and code '+str(rule))
    if not isinstance(rule,list): rule = [rule]
    self.makeRules[name] = [dependencies,rule]
    return

  def addMakeMacro(self, name, value):
    '''Designate that "name" should be defined to "value" in the makefile header (bmake file)'''
    self.logPrint('Defined make macro "'+name+'" to "'+str(value)+'"')
    self.makeMacros[name] = value
    return

  def getMakeMacro(self, name):
    return self.makeMacros.get(name)

  def delMakeMacro(self, name):
    '''Designate that "name" should be deleted (never put in) configuration header'''
    self.logPrint('Deleting "'+name+'"')
    if name in self.makeMacros: del self.makeMacros[name]
    return

  def addDefine(self, name, value):
    '''Designate that "name" should be defined to "value" in the configuration header'''
    self.logPrint('Defined "'+name+'" to "'+str(value)+'"')
    self.defines[name] = value
    return

  def delDefine(self, name):
    '''Designate that "name" should be deleted (never put in)  configuration header'''
    self.logPrint('Deleting "'+name+'"')
    if name in self.defines: del self.defines[name]
    return

  def addTypedef(self, name, value):
    '''Designate that "name" should be typedefed to "value" in the configuration header'''
    self.logPrint('Typedefed "'+name+'" to "'+str(value)+'"')
    self.typedefs[value] = name
    return

  def addPrototype(self, prototype, language = 'All'):
    '''Add a missing function prototype
       - The language argument defaults to "All"
       - Other language choices are C, Cxx, extern C'''
    self.logPrint('Added prototype '+prototype+' to language '+language)
    language = language.replace('+', 'x')
    if not language in self.prototypes:
      self.prototypes[language] = []
    self.prototypes[language].append(prototype)
    return

  def addSubstitution(self, name, value):
    '''Designate that "@name@" should be replaced by "value" in all files which experience substitution'''
    self.logPrint('Substituting "'+name+'" with "'+str(value)+'"')
    self.subst[name] = value
    return

  def addArgumentSubstitution(self, name, arg):
    '''Designate that "@name@" should be replaced by "arg" in all files which experience substitution'''
    self.logPrint('Substituting "'+name+'" with '+str(arg)+'('+str(self.argDB[arg])+')')
    self.argSubst[name] = arg
    return

  ################
  # Program Checks
  def checkExecutable(self, dir, name):
    prog  = os.path.join(dir, name)
    # also strip any \ before spaces, braces, so that we can specify paths the way we want them in makefiles.
    prog  = prog.replace('\ ',' ').replace('\(','(').replace('\)',')')
    found = 0
    self.logWrite('Checking for program '+prog+'...')
    if os.path.isfile(prog) and os.access(prog, os.X_OK):
      found = 1
      self.logWrite('found\n')
    else:
      self.logWrite('not found\n')
    return found

  def getExecutable(self, names, path = [], getFullPath = 0, useDefaultPath = 0, resultName = '', setMakeMacro = 1):
    '''Search for an executable in the list names
       - Each name in the list is tried for each entry in the path until a name is located, then it stops
       - If found, the path is stored in the variable "name", or "resultName" if given
       - By default, a make macro "resultName" will hold the path'''
    found = 0
    if isinstance(names,str) and names.startswith('/'):
      path = os.path.dirname(names)
      names = os.path.basename(names)

    if isinstance(names, str):
      names = [names]
    if isinstance(path, str):
      path = path.split(os.path.pathsep)
    if not len(path):
      useDefaultPath = 1

    def getNames(name, resultName):
      import re
      prog = re.match(r'(.*?)(?<!\\)(\s.*)',name)
      if prog:
        name = prog.group(1)
        options = prog.group(2)
      else:
        options = ''
      if not resultName:
        varName = name
      else:
        varName = resultName
      return name, options, varName

    varName = names[0]
    varPath = ''
    for d in path:
      for name in names:
        name, options, varName = getNames(name, resultName)
        if self.checkExecutable(d, name):
          found = 1
          getFullPath = 1
          varPath = d
          break
      if found: break
    if useDefaultPath and not found:
      for d in os.environ['PATH'].split(os.path.pathsep):
        for name in names:
          name, options, varName = getNames(name, resultName)
          if self.checkExecutable(d, name):
            found = 1
            varPath = d
            break
        if found: break
    if not found:
      dirs = self.argDB['with-executables-search-path']
      if not isinstance(dirs, list): dirs = [dirs]
      for d in dirs:
        for name in names:
          name, options, varName = getNames(name, resultName)
          if self.checkExecutable(d, name):
            found = 1
            getFullPath = 1
            varPath = d
            break
        if found: break

    if found:
      if getFullPath:
        setattr(self, varName, os.path.abspath(os.path.join(varPath, name))+options)
      else:
        setattr(self, varName, name+options)
      if setMakeMacro:
        self.addMakeMacro(varName.upper(), getattr(self, varName))
    else:
      def logPrintFilesInPath(path):
        for d in path:
          try:
            self.logWrite('      '+str(os.listdir(d))+'\n')
          except OSError as e:
            self.logWrite('      Warning accessing '+d+' gives errors: '+str(e)+'\n')
        return
      self.logWrite('  Unable to find programs '+str(names)+' providing listing of each search directory to help debug\n')
      self.logWrite('    Path provided in Python program\n')
      logPrintFilesInPath(path)
      if useDefaultPath:
        if os.environ['PATH'].split(os.path.pathsep):
          self.logWrite('    Path provided by default path\n')
          logPrintFilesInPath(os.environ['PATH'].split(os.path.pathsep))
      dirs = self.argDB['with-executables-search-path']
      if not isinstance(dirs, list): dirs = [dirs]
      if dirs:
        self.logWrite('    Path provided by --with-executables-search-path\n')
        logPrintFilesInPath(dirs)
    return found

  def getExecutables(self, names, path = '', getFullPath = 0, useDefaultPath = 0, resultName = ''):
    '''Search for an executable in the list names
       - The full path given is searched for each name in turn
       - If found, the path is stored in the variable "name", or "resultName" if given'''
    for name in names:
      if self.getExecutable(name, path = path, getFullPath = getFullPath, useDefaultPath = useDefaultPath, resultName = resultName):
        return name
    return None

  ###############################################
  # Preprocessor, Compiler, and Linker Operations
  def pushLanguage(self, language):
    if language == 'C++': language = 'Cxx'
    self.language.append(language)
    return self.language[-1]

  def popLanguage(self):
    self.language.pop()
    return self.language[-1]

  @contextlib.contextmanager
  def Language(self, lang):
    if lang is None:
      yield
    else:
      self.pushLanguage(lang)
      yield
      self.popLanguage()

  def getHeaders(self):
    self.compilerDefines = os.path.join(self.tmpDir, 'confdefs.h')
    self.compilerFixes   = os.path.join(self.tmpDir, 'conffix.h')
    return

  def getPreprocessor(self):
    self.getHeaders()
    preprocessor       = self.framework.getPreprocessorObject(self.language[-1])
    preprocessor.checkSetup()
    return preprocessor.getProcessor()

  def getCompiler(self, lang=None):
    with self.Language(lang):
      self.getHeaders()
      compiler            = self.framework.getCompilerObject(self.language[-1])
      compiler.checkSetup()
      self.compilerSource = os.path.join(self.tmpDir, 'conftest'+compiler.sourceExtension)
      self.compilerObj    = os.path.join(self.tmpDir, compiler.getTarget(self.compilerSource))
      return compiler.getProcessor()

  def getCompilerFlags(self):
    return self.framework.getCompilerObject(self.language[-1]).getFlags()

  def getLinker(self):
    self.getHeaders()
    linker            = self.framework.getLinkerObject(self.language[-1])
    linker.checkSetup()
    self.linkerSource = os.path.join(self.tmpDir, 'conftest'+linker.sourceExtension)
    self.linkerObj    = linker.getTarget(self.linkerSource, 0)
    return linker.getProcessor()

  def getLinkerFlags(self):
    return self.framework.getLinkerObject(self.language[-1]).getFlags()

  def getSharedLinker(self):
    self.getHeaders()
    linker            = self.framework.getSharedLinkerObject(self.language[-1])
    linker.checkSetup()
    self.linkerSource = os.path.join(self.tmpDir, 'conftest'+linker.sourceExtension)
    self.linkerObj    = linker.getTarget(self.linkerSource, 1)
    return linker.getProcessor()

  def getSharedLinkerFlags(self):
    return self.framework.getSharedLinkerObject(self.language[-1]).getFlags()

  def getDynamicLinker(self):
    self.getHeaders()
    linker            = self.framework.getDynamicLinkerObject(self.language[-1])
    linker.checkSetup()
    self.linkerSource = os.path.join(self.tmpDir, 'conftest'+linker.sourceExtension)
    self.linkerObj    = linker.getTarget(self.linkerSource, 1)
    return linker.getProcessor()

  def getDynamicLinkerFlags(self):
    return self.framework.getDynamicLinkerObject(self.language[-1]).getFlags()

  def getPreprocessorCmd(self):
    self.getCompiler()
    preprocessor = self.framework.getPreprocessorObject(self.language[-1])
    preprocessor.checkSetup()
    preprocessor.includeDirectories.add(self.tmpDir)
    return preprocessor.getCommand(self.compilerSource)

  def getCompilerCmd(self):
    self.getCompiler()
    compiler = self.framework.getCompilerObject(self.language[-1])
    compiler.checkSetup()
    compiler.includeDirectories.add(self.tmpDir)
    return compiler.getCommand(self.compilerSource, self.compilerObj)

  def getLinkerCmd(self):
    self.getLinker()
    linker = self.framework.getLinkerObject(self.language[-1])
    linker.checkSetup()
    return linker.getCommand(self.linkerSource, self.linkerObj)

  def getFullLinkerCmd(self, objects, executable):
    self.getLinker()
    linker = self.framework.getLinkerObject(self.language[-1])
    linker.checkSetup()
    return linker.getCommand(objects, executable)

  def getSharedLinkerCmd(self):
    self.getSharedLinker()
    linker = self.framework.getSharedLinkerObject(self.language[-1])
    linker.checkSetup()
    return linker.getCommand(self.linkerSource, self.linkerObj)

  def getDynamicLinkerCmd(self):
    self.getDynamicLinker()
    linker = self.framework.getDynamicLinkerObject(self.language[-1])
    linker.checkSetup()
    return linker.getCommand(self.linkerSource, self.linkerObj)

  def getCode(self, includes, body = None, codeBegin = None, codeEnd = None):
    language = self.language[-1]
    if includes and not includes[-1] == '\n':
      includes += '\n'
    if language in ['C', 'CUDA', 'Cxx', 'HIP', 'SYCL']:
      codeStr = ''
      if self.compilerDefines: codeStr = '#include "'+os.path.basename(self.compilerDefines)+'"\n'
      codeStr += '#include "conffix.h"\n'+includes
      if not body is None:
        if codeBegin is None:
          codeBegin = '\nint main() {\n'
        if codeEnd is None:
          codeEnd   = ';\n  return 0;\n}\n'
        codeStr += codeBegin+body+codeEnd
    elif language == 'FC':
      if not includes is None and body is None:
        codeStr = includes
      else:
        codeStr = ''
      if not body is None:
        if codeBegin is None:
          codeBegin = '      program main\n'
          if not includes is None:
            codeBegin = codeBegin+includes
        if codeEnd is None:
          codeEnd   = '\n      end\n'
        codeStr += codeBegin+body+codeEnd
    else:
      raise RuntimeError('Cannot determine code body for language: '+language)
    return codeStr

  def preprocess(self, codeStr, timeout = 600.0):
    def report(command, status, output, error):
      if error or status:
        self.logError('preprocessor', status, output, error)
        self.logWrite('Source:\n'+self.getCode(codeStr))

    command = self.getPreprocessorCmd()
    if self.compilerDefines: self.framework.outputHeader(self.compilerDefines)
    self.framework.outputCHeader(self.compilerFixes)
    self.logWrite('Preprocessing source:\n'+self.getCode(codeStr))
    f = open(self.compilerSource, 'w')
    f.write(self.getCode(codeStr))
    f.close()
    (out, err, ret) = Configure.executeShellCommand(command, checkCommand = report, timeout = timeout, log = self.log, logOutputflg = False, lineLimit = 100000)
    if self.cleanup:
      for filename in [self.compilerDefines, self.compilerFixes, self.compilerSource]:
        if os.path.isfile(filename): os.remove(filename)
    return (out, err, ret)

  def outputPreprocess(self, codeStr):
    '''Return the contents of stdout when preprocessing "codeStr"'''
    return self.preprocess(codeStr)[0]

  def checkPreprocess(self, codeStr, timeout = 600.0):
    '''Return True if no error occurred
       - An error is signaled by a nonzero return code, or output on stderr'''
    (out, err, ret) = self.preprocess(codeStr, timeout = timeout)
    err = self.framework.filterPreprocessOutput(err, self.log)
    return not ret and not len(err)

  # Should be static
  def getPreprocessorFlagsName(self, language):
    if language == 'C':
      flagsArg = 'CPPFLAGS'
    elif language == 'CUDA':
      flagsArg = 'CUDAPPFLAGS'
    elif language == 'Cxx':
      flagsArg = 'CXXPPFLAGS'
    elif language == 'FC':
      flagsArg = 'FPPFLAGS'
    elif language == 'HIP':
      flagsArg = 'HIPPPFLAGS'
    elif language == 'SYCL':
      flagsArg = 'SYCLPPFLAGS'
    else:
      raise RuntimeError('Unknown language: '+language)
    return flagsArg

  def getPreprocessorFlagsArg(self):
    '''Return the name of the argument which holds the preprocessor flags for the current language'''
    return self.getPreprocessorFlagsName(self.language[-1])

  def filterCompileOutput(self, output):
    return self.framework.filterCompileOutput(output)

  def outputCompile(self, includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None):
    '''Return the error output from this compile and the return code'''
    def report(command, status, output, error):
      if error or status:
        self.logError('compiler', status, output, error)
      else:
        self.logWrite('Successful compile:\n')
      self.logWrite('Source:\n'+self.getCode(includes, body, codeBegin, codeEnd))

    cleanup = cleanup and self.framework.doCleanup
    command = self.getCompilerCmd()
    if self.compilerDefines: self.framework.outputHeader(self.compilerDefines)
    self.framework.outputCHeader(self.compilerFixes)
    f = open(self.compilerSource, 'w')
    f.write(self.getCode(includes, body, codeBegin, codeEnd))
    f.close()
    (out, err, ret) = Configure.executeShellCommand(command, checkCommand = report, log = self.log)
    if not os.path.isfile(self.compilerObj):
      err += '\nPETSc Error: No output file produced'
    if cleanup:
      for filename in [self.compilerDefines, self.compilerFixes, self.compilerSource, self.compilerObj]:
        if os.path.isfile(filename): os.remove(filename)
    return (out, err, ret)

  def checkCompile(self, includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None):
    '''Returns True if the compile was successful'''
    (output, error, returnCode) = self.outputCompile(includes, body, cleanup, codeBegin, codeEnd)
    output = self.filterCompileOutput(output+'\n'+error)
    return not (returnCode or len(output))

  def getCompilerFlagsName(language, compilerOnly = 0):
    if language == 'C':
      flagsArg = 'CFLAGS'
    elif language == 'CUDA':
      flagsArg = 'CUDAFLAGS'
    elif language == 'Cxx':
      if compilerOnly:
        flagsArg = 'CXX_CXXFLAGS'
      else:
        flagsArg = 'CXXFLAGS'
    elif language == 'HIP':
      flagsArg = 'HIPCCFLAGS'
    elif language == 'SYCL':
      flagsArg = 'SYCLCXXFLAGS'
    elif language == 'FC':
      flagsArg = 'FFLAGS'
    else:
      raise RuntimeError('Unknown language: '+language)
    return flagsArg
  getCompilerFlagsName = staticmethod(getCompilerFlagsName)

  def getCompilerFlagsArg(self, compilerOnly = 0):
    '''Return the name of the argument which holds the compiler flags for the current language'''
    return self.getCompilerFlagsName(self.language[-1], compilerOnly)

  def filterLinkOutput(self, output):
    return self.framework.filterLinkOutput(output)

  def outputLink(self, includes, body, cleanup = 1, codeBegin = None, codeEnd = None, shared = 0, linkLanguage=None, examineOutput=lambda ret,out,err:None):
    import sys

    (out, err, ret) = self.outputCompile(includes, body, cleanup = 0, codeBegin = codeBegin, codeEnd = codeEnd)
    examineOutput(ret, out, err)
    out = self.filterCompileOutput(out+'\n'+err)
    if ret or len(out):
      self.logPrint('Compile failed inside link\n'+out)
      self.linkerObj = ''
      return (out, ret)

    cleanup = cleanup and self.framework.doCleanup

    langPushed = 0
    if linkLanguage is not None and linkLanguage != self.language[-1]:
      self.pushLanguage(linkLanguage)
      langPushed = 1
    if shared == 'dynamic':
      cmd = self.getDynamicLinkerCmd()
    elif shared:
      cmd = self.getSharedLinkerCmd()
    else:
      cmd = self.getLinkerCmd()
    if langPushed:
      self.popLanguage()

    linkerObj = self.linkerObj
    def report(command, status, output, error):
      if error or status:
        self.logError('linker', status, output, error)
        examineOutput(status, output, error)
      return
    (out, err, ret) = Configure.executeShellCommand(cmd, checkCommand = report, log = self.log)
    self.linkerObj = linkerObj
    if os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    if cleanup:
      if os.path.isfile(self.linkerObj):os.remove(self.linkerObj)
      pdbfile = os.path.splitext(self.linkerObj)[0]+'.pdb'
      if os.path.isfile(pdbfile): os.remove(pdbfile)
    return (out+'\n'+err, ret)

  def checkLink(self, includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None, shared = 0, linkLanguage=None, examineOutput=lambda ret,out,err:None):
    (output, returnCode) = self.outputLink(includes, body, cleanup, codeBegin, codeEnd, shared, linkLanguage, examineOutput)
    output = self.filterLinkOutput(output)
    return not (returnCode or len(output))

  def getLinkerFlagsName(language):
    if language in ['C', 'CUDA', 'Cxx', 'FC', 'HIP', 'SYCL']:
      flagsArg = 'LDFLAGS'
    else:
      raise RuntimeError('Unknown language: '+language)
    return flagsArg
  getLinkerFlagsName = staticmethod(getLinkerFlagsName)

  def getLinkerFlagsArg(self):
    '''Return the name of the argument which holds the linker flags for the current language'''
    return self.getLinkerFlagsName(self.language[-1])

  def outputRun(self, includes, body, cleanup = 1, defaultOutputArg = '', executor = None,linkLanguage=None, timeout = 60, threads = 1):
    if not self.checkLink(includes, body, cleanup = 0, linkLanguage=linkLanguage): return ('', 1)
    self.logWrite('Testing executable '+self.linkerObj+' to see if it can be run\n')
    if not os.path.isfile(self.linkerObj):
      self.logWrite('ERROR executable '+self.linkerObj+' does not exist\n')
      return ('', 1)
    if not os.access(self.linkerObj, os.X_OK):
      self.logWrite('ERROR while running executable: '+self.linkerObj+' is not executable\n')
      return ('', 1)
    if self.argDB['with-batch']:
      if defaultOutputArg:
        if defaultOutputArg in self.argDB:
          return (self.argDB[defaultOutputArg], 0)
        else:
          raise ConfigureSetupError('Must give a default value for '+defaultOutputArg+' since generated executables cannot be run with the --with-batch option')
      else:
        raise ConfigureSetupError('Generated executables cannot be run with the --with-batch option')
    cleanup = cleanup and self.framework.doCleanup
    if executor:
      command = executor+' '+self.linkerObj
    else:
      command = self.linkerObj
    output  = ''
    error   = ''
    status  = 1
    self.logWrite('Executing: '+command+'\n')
    try:
      (output, error, status) = Configure.executeShellCommand(command, log = self.log, timeout = timeout, threads = threads)
    except RuntimeError as e:
      self.logWrite('ERROR while running executable: '+str(e)+'\n')
      if str(e).find('Runaway process exceeded time limit') > -1:
        raise RuntimeError('Runaway process exceeded time limit')
    if os.path.isfile(self.compilerObj):
      try:
        os.remove(self.compilerObj)
      except RuntimeError as e:
        self.logWrite('ERROR while removing object file: '+str(e)+'\n')
    if cleanup and os.path.isfile(self.linkerObj):
      try:
        if os.path.exists('/usr/bin/cygcheck.exe'): time.sleep(1)
        os.remove(self.linkerObj)
      except RuntimeError as e:
        self.logWrite('ERROR while removing executable file: '+str(e)+'\n')
    return (output+error, status)

  def checkRun(self, includes = '', body = '', cleanup = 1, defaultArg = '', executor = None, linkLanguage=None, timeout = 60, threads = 1):
    (output, returnCode) = self.outputRun(includes, body, cleanup, defaultArg, executor,linkLanguage=linkLanguage, timeout = timeout, threads = threads)
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
          for ext in ['a', 'so','dylib']:
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

  def setupPackageDependencies(self, framework):
    '''All calls to the framework addPackageDependency() should be made here'''
    pass

  def setupDependencies(self, framework):
    '''All calls to the framework require() should be made here'''
    self.framework = framework

  def configure(self):
    pass

  def no_configure(self):
    pass
