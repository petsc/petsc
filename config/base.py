import commands
import os
import re
import select

class Configure:
  def __init__(self, framework):
    self.framework = framework
    self.defines   = {}
    self.subst     = {}
    self.argSubst  = {}
    # Interaction with Autoconf
    self.m4           = '/usr/bin/m4'
    self.acMacroDir   = '/usr/share/autoconf'
    self.acLocalDir   = 'config'
    self.acReload     = '--reload'
    self.acMsgFD      = '2'
    self.configAuxDir = 'config'
    # Interaction with the shell
    self.shell = '/bin/sh'
    # Preprocessing, compiling, and linking
    self.language = []
    self.pushLanguage('C')
    return

  def getAcCCFD(self):
    return str(self.log.fileno())

  def getRoot(self):
    import sys
    # This has the problem that when we reload a module of the same name, this gets screwed up
    #   Therefore, we call it in the initializer, and stash it
    if not hasattr(self, '_root_'):
      if hasattr(sys.modules[self.__module__], '__file__'):
        self._root_ = os.path.abspath(os.path.dirname(sys.modules[self.__module__].__file__))
      else:
        self._root_ = os.getcwd()
    return self._root_

  def executeTest(self, test, args = []):
    self.framework.log.write('================================================================================\n')
    self.framework.log.write('TEST '+str(test.im_func.__name__)+' from '+str(test.im_class.__module__)+'\n')
    if test.__doc__: self.framework.log.write('  '+test.__doc__+'\n')
    if not isinstance(args, list): args = [args]
    return apply(test, args)

  #################################
  # Define and Substitution Support
  def addDefine(self, name, value):
    '''Designate that "name" should be defined to "value" in the configuration header'''
    self.defines[name] = value
    return

  def addSubstitution(self, name, value):
    '''Designate that "@name@" should be replaced by "value" in all files which experience substitution'''
    self.subst[name] = value
    return

  def addArgumentSubstitution(self, name, arg):
    '''Designate that "@name@" should be replaced by "arg" in all files which experience substitution'''
    self.argSubst[name] = arg
    return

  ################
  # Program Checks
  def getExecutable(self, name, path = '', getFullPath = 0, resultName = ''):
    if not path or path[-1] == ':': path += os.environ['PATH']
    if not resultName: resultName = name
    found = 0
    for dir in path.split(':'):
      prog = os.path.join(dir, name)

      self.framework.log.write('Checking for program '+prog+'...')
      if os.path.isfile(prog) and os.access(prog, os.X_OK):
        if getFullPath:
          setattr(self, resultName, os.path.abspath(prog))
        else:
          setattr(self, resultName, name)
        self.addSubstitution(resultName.upper(), getattr(self, resultName))
        found = 1
        self.framework.log.write('found\n')
        break
      self.framework.log.write('not found\n')
    return found

  def getExecutables(self, names, path = '', getFullPath = 0, resultName = ''):
    for name in names:
      if self.getExecutable(name, path, getFullPath, resultName):
        return name
    return None

  ###############################################
  # Preprocessor, Compiler, and Linker Operations
  def pushLanguage(self, language):
    self.language.append(language)
    return self.setLanguage(self.language[-1])

  def popLanguage(self):
    self.language.pop()
    return self.setLanguage(self.language[-1])

  def setLanguage(self, language):
    if language == 'C':
      self.compilerDefines = 'confdefs.h'
    elif language == 'C++':
      self.compilerDefines = 'confdefs.h'
    elif language == 'F77':
      self.compilerDefines = 'confdefs.h'
    else:
      raise RuntimeError('Unknown language: '+language)
    return

  def checkCCompilerSetup(self):
    if not self.framework.argDB.has_key('CC'):
      raise RuntimeError('Could not find a C compiler. Make sure the compiler module is loaded.')
    if not self.framework.argDB.has_key('CPP'):
      raise RuntimeError('Could not find a C preprocessor. Make sure the compiler module is loaded.')
    return

  def checkCxxCompilerSetup(self):
    if not self.framework.argDB.has_key('CXX'):
      raise RuntimeError('Could not find a C++ compiler. Make sure the compiler module is loaded.')
    if not self.framework.argDB.has_key('CXXCPP'):
      raise RuntimeError('Could not find a C++ preprocessor. Make sure the compiler module is loaded.')
    return

  def checkF77CompilerSetup(self):
    if not self.framework.argDB.has_key('FC'):
      raise RuntimeError('Could not find a Fortran 77 compiler. Make sure the compiler module is loaded.')
    return

  def getCompiler(self):
    language = self.language[-1]
    if language == 'C':
      self.checkCCompilerSetup()
      self.compilerName   = 'CC'
      self.compilerSource = 'conftest.c'
      self.compilerObj    = 'conftest.o'
    elif language == 'C++':
      self.checkCxxCompilerSetup()
      self.compilerName   = 'CXX'
      self.compilerSource = 'conftest.cc'
      self.compilerObj    = 'conftest.o'
    elif language == 'F77':
      self.checkF77CompilerSetup()
      self.compilerName   = 'FC'
      self.compilerSource = 'conftest.f'
      self.compilerObj    = 'conftest.o'
    else:
      raise RuntimeError('Unknown language: '+language)
    self.compiler = self.framework.argDB[self.compilerName]
    return self.compiler

  def getCppCmd(self):
    language = self.language[-1]
    self.getCompiler()
    if language == 'C':
      self.cpp      = self.framework.argDB['CPP']
      self.cppFlags = self.framework.argDB['CPPFLAGS']
      self.cppCmd   = self.cpp+' '+self.cppFlags+' '+self.compilerSource
    elif language == 'C++':
      self.cpp      = self.framework.argDB['CXXCPP']
      self.cppFlags = self.framework.argDB['CPPFLAGS']
      self.cppCmd   = self.cpp+' '+self.cppFlags+' '+self.compilerSource
    elif language == 'F77':
      self.cpp      = self.framework.argDB['CPP']
      self.cppFlags = self.framework.argDB['CPPFLAGS']
      self.cppCmd   = self.cpp+' '+self.cppFlags+' '+self.compilerSource
    else:
      raise RuntimeError('Unknown language: '+language)
    return self.cppCmd

  def getCompilerCmd(self):
    language = self.language[-1]
    self.getCompiler()
    if language == 'C':
      self.compilerFlags   = self.framework.argDB['CFLAGS']+' '+self.framework.argDB['CPPFLAGS']
      self.compilerCmd     = self.compiler+' -c -o '+self.compilerObj+' '+self.compilerFlags+' '+self.compilerSource
    elif language == 'C++':
      self.compilerFlags   = self.framework.argDB['CXXFLAGS']+' '+self.framework.argDB['CPPFLAGS']
      self.compilerCmd     = self.compiler+' -c -o '+self.compilerObj+' '+self.compilerFlags+' '+self.compilerSource
    elif language == 'F77':
      self.compilerFlags  = self.framework.argDB['FFLAGS']
      self.compilerCmd    = self.compiler+' -c -o '+self.compilerObj+' '+self.compilerFlags+' '+self.compilerSource
    else:
      raise RuntimeError('Unknown language: '+language)
    return self.compilerCmd

  def getLinkerCmd(self):
    language = self.language[-1]
    self.getCompiler()
    if language == 'C':
      self.linker      = self.compiler
      self.linkerObj   = 'conftest'
      self.linkerFlags = self.framework.argDB['CFLAGS']+' '+self.framework.argDB['CPPFLAGS']+' '+self.framework.argDB['LDFLAGS']
      self.linkerCmd   = self.linker+' -o '+self.linkerObj+' '+self.linkerFlags+' conftest.o '+self.framework.argDB['LIBS']
    elif language == 'C++':
      self.linker      = self.compiler
      self.linkerObj   = 'conftest'
      self.linkerFlags = self.framework.argDB['CXXFLAGS']+' '+self.framework.argDB['CPPFLAGS']+' '+self.framework.argDB['LDFLAGS']
      self.linkerCmd   = self.linker+' -o '+self.linkerObj+' '+self.linkerFlags+' conftest.o '+self.framework.argDB['LIBS']
    elif language == 'F77':
      self.linker      = self.compiler
      self.linkerObj   = 'conftest'
      self.linkerFlags = self.framework.argDB['FFLAGS']+' '+self.framework.argDB['LDFLAGS']
      self.linkerCmd   = self.linker+' -o '+self.linkerObj+' '+self.linkerFlags+' conftest.o '+self.framework.argDB['LIBS']
    else:
      raise RuntimeError('Unknown language: '+language)
    return self.linkerCmd

  def getCode(self, includes, body = None):
    language = self.language[-1]
    if includes and not includes[-1] == '\n':
      includes += '\n'
    if language == 'C' or language == 'C++':
      codeStr = '#include "confdefs.h"\n'+includes
      if not body is None:
        codeStr += '\nint main() {\n'+body+';\n  return 0;\n}\n'
    elif language == 'F77':
      if not body is None:
        codeStr = '      program main\n'+body+'\n      end\n'
      else:
        codeStr = includes
    else:
      raise RuntimeError('Invalid language: '+language)
    return codeStr

  def openPipe(self, command):
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

  # TODO: Make selecting between ouput and err work (not robust right now), then we can combine the functions
  def outputPreprocess(self, codeStr):
    command = self.getCppCmd()
    self.framework.outputHeader(self.compilerDefines)
    f = file(self.compilerSource, 'w')
    f.write(self.getCode(codeStr))
    f.close()
    self.framework.log.write('Executing: '+command+'\n')
    (input, output, err, pipe) = self.openPipe(command)
    input.close()
    out   = ''
    ready = select.select([output, err], [], [])
    if len(ready[0]) and output in ready[0]:
      out = ready[0][0].read()
    elif len(ready[0]):
      self.framework.log.write('ERR (preprocessor): '+ready[0][0].read())
      self.framework.log.write('Source:\n'+self.getCode(codeStr))
    err.close()
    output.close()
    if os.path.isfile(self.compilerDefines): os.remove(self.compilerDefines)
    if os.path.isfile(self.compilerSource): os.remove(self.compilerSource)
    return out

  def checkPreprocess(self, codeStr):
    command = self.getCppCmd()
    self.framework.outputHeader(self.compilerDefines)
    f = file(self.compilerSource, 'w')
    f.write(self.getCode(codeStr))
    f.close()
    self.framework.log.write('Executing: '+command+'\n')
    (input, output, err, pipe) = self.openPipe(command)
    input.close()
    out   = ''
    ready = select.select([output, err], [], [])
    if len(ready[0]) and err in ready[0]:
      # Log failure of preprocessor
      out = err.read()
      if out:
        self.framework.log.write('ERR (preprocessor): '+out)
        self.framework.log.write('Source:\n'+self.getCode(codeStr))
    else:
      for fd in ready[0]: fd.read()
    err.close()
    output.close()
    ret = None
    if pipe:
      ret = pipe.wait()
    if os.path.isfile(self.compilerDefines): os.remove(self.compilerDefines)
    if os.path.isfile(self.compilerSource): os.remove(self.compilerSource)
    return not ret or not len(out)

  def outputCompile(self, includes = '', body = '', cleanup = 1):
    command = self.getCompilerCmd()
    self.framework.outputHeader(self.compilerDefines)
    f = file(self.compilerSource, 'w')
    f.write(self.getCode(includes, body))
    f.close()
    self.framework.log.write('Executing: '+command+'\n')
    (input, output, err, pipe) = self.openPipe(command)
    input.close()
    ret = None
    if pipe:
      ret = pipe.wait()
    out   = ''
    ready = select.select([err], [], [], 0.1)
    if len(ready[0]):
      # Log failure of compiler
      out = ready[0][0].read()
    if ret and not out:
      out = str(ret)
    if out:
      self.framework.log.write('ERR (compiler): '+out)
      self.framework.log.write('ret = '+str(ret))
      self.framework.log.write('Source:\n'+self.getCode(includes, body))
    err.close()
    output.close()
    if os.path.isfile(self.compilerDefines): os.remove(self.compilerDefines)
    if os.path.isfile(self.compilerSource): os.remove(self.compilerSource)
    if cleanup and os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    return out

  def checkCompile(self, includes = '', body = '', cleanup = 1):
    return not len(self.outputCompile(includes, body, cleanup))

  def outputLink(self, includes, body, cleanup = 1):
    import sys

    out = self.outputCompile(includes, body, cleanup = 0)
    if len(out): return out
    command = self.getLinkerCmd()
    self.framework.log.write('Executing: '+command+'\n')
    (input, output, err, pipe) = self.openPipe(command)
    input.close()
    ret = None
    if pipe:
      ret = pipe.wait()
    out   = ''
    ready = select.select([err], [], [], 0.1)
    if len(ready[0]):
      # Log failure of linker
      out = ready[0][0].read()
    if ret and not out:
      out = str(ret)
    if out:
      self.framework.log.write('ERR (linker): '+out)
      self.framework.log.write('ret = '+str(ret))
      self.framework.log.write(' in '+self.getLinkerCmd()+'\n')
    err.close()
    output.close()
    if sys.platform[:3] == 'win' or sys.platform == 'cygwin':
      self.linkerObj = self.linkerObj+'.exe'
    if os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    if cleanup and os.path.isfile(self.linkerObj): os.remove(self.linkerObj)
    return out

  def checkLink(self, includes, body, cleanup = 1):
    return not len(self.outputLink(includes, body, cleanup))

  def checkRun(self, includes, body):
    if not self.checkLink(includes, body, cleanup = 0): return 0
    success = 0
    if not os.path.isfile(self.linkerObj) or not os.access(self.linkerObj, os.X_OK):
      self.framework.log.write('ERR (executable): '+self.linkerObj+' is not executable')
      return success
    command = './'+self.linkerObj
    self.framework.log.write('Executing: '+command+'\n')
    (status, output) = commands.getstatusoutput(command)
    if not status:
      success = 1
    else:
      self.framework.log.write('ERR (executable): '+output)
    if os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    if os.path.isfile(self.linkerObj): os.remove(self.linkerObj)
    return success

  ######################################
  # Methods for Autoconf Macro Execution
  def getDefaultMacros(self):
    '''Macros that seems necessary to run any given Autoconf macro'''
    return 'AC_INIT_BINSH\nAC_CONFIG_AUX_DIR('+self.configAuxDir+')\n'

  def getMacroVersion(self, macro):
    '''This is the version of Autoconf required by the macro'''
    m = re.search(r'^dnl\s+Version:\s+(?P<version>\d+\.\d+)', macro, re.M)
    if m:
      return m.group('version')
    else:
      return ''

  def getMacroVariables(self, macro):
    '''These are the variables output by the macro'''
    varRE = re.compile(r'^dnl\s+Variable:\s+(?P<variable>\w+)', re.M)
    return varRE.findall(macro)

  def replaceDefaultDescriptors(self, codeStr):
    '''Autoconf defines several default file descriptors, which we must assign'''
    newCode = re.sub('AC_FD_MSG', self.acMsgFD, codeStr)
    newCode = re.sub('AC_FD_CC',  self.getAcCCFD(),  newCode)
    return newCode

  def findUndefinedMacros(self, codeStr):
    '''This finds Auotconf macros which have not been expanded because no definitions have been found'''
    matches = re.findall(r'AC_\w+', codeStr)
    if len(matches):
      msg = 'Undefined macros:\n'
      for m in matches: msg += '  '+m+'\n'
      raise RuntimeError(msg)
    return

  def macroToShell(self, macro):
    '''This takes the text of an Autoconf macro and returns a tuple of the corresponding shell code and output variable names'''
    self.getMacroVersion(macro)
    command = self.m4
    if self.acMacroDir:
      command += ' -I'+self.acMacroDir
    if self.acLocalDir:
      command += ' -I'+self.acLocalDir+' -DAC_LOCALDIR='+self.acLocalDir
    if self.acReload and os.path.exists(os.path.join(self.acMacroDir, 'autoconf.m4f')):
      command += ' '+self.acReload+' autoconf.m4f'
    else:
      command += ' autoconf.m4'
    (input, output) = os.popen2(command)
    input.write(self.getDefaultMacros()+macro)
    input.close()
    out = output.read()
    shellCode = self.replaceDefaultDescriptors(out)
    self.findUndefinedMacros(shellCode)
    output.close()
    return (re.sub('__oline__', '0', shellCode), self.getMacroVariables(macro))

  def getDefaultVariables(self):
    '''These shell variables are set by Autoconf, and seem to be necessary to run any given macro'''
    return '''
    host=NONE
    nonopt=NONE
    CONFIG_SHELL=%s
    ac_ext="c"
    ac_exeext=""
    ac_cpp=\'$CPP $CPPFLAGS\'
    ac_compile=\'${CC-cc} -c $CFLAGS $CPPFLAGS conftest.$ac_ext 1>&%s\'
    ac_link=\'${CC-cc} -o conftest${ac_exeext} $CFLAGS $CPPFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&%s\'
    exec %s>>%s
    ''' % (self.shell, self.getAcCCFD(), self.getAcCCFD(), self.getAcCCFD(), self.framework.logName)

  def parseShellOutput(self, output):
    '''This retrieves the output variable values from macro shell code'''
    results = {}
    varRE   = re.compile(r'(?P<name>\w+)\s+=\s+(?P<value>.*)')
    for line in output.split('\n'):
      m = varRE.match(line)
      if m: results[m.group('name')] = m.group('value')
    return results

  def executeShellCode(self, code):
    '''This executes the shell code for an Autoconf macro, appending code which causes the output variables to be printed'''
    codeStr  = self.getDefaultVariables()
    codeStr += code[0]
    for var in code[1]:
      codeStr += 'echo "'+var+' = " ${'+var+'}\n'
    self.framework.outputHeader(self.compilerDefines)
    (input, output) = os.popen4(self.shell)
    input.write(codeStr)
    input.close()
    results = output.read()
    output.close()
    if os.path.isfile(self.compilerDefines): os.remove(self.compilerDefines)
    return self.parseShellOutput(results)

  def configure(self):
    pass
