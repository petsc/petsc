#!/usr/bin/env python
import PETSc
import PETSc.Configure
import nargs

import commands
import cPickle
import os
import re
import select
import sys

## SECTION: Initialization
# Set default language to C

## SECTION: Installation

class Configure:
  def __init__(self, framework):
    self.framework = framework
    self.defines   = {}
    self.subst     = {}
    self.help      = {}
    # Interaction with Autoconf
    self.m4           = '/usr/bin/m4'
    self.acMacroDir   = '/usr/share/autoconf'
    self.acLocalDir   = 'config'
    self.acReload     = '--reload'
    self.acMsgFD      = '2'
    self.acCCFD       = str(self.framework.log.fileno())
    self.configAuxDir = 'config'
    # Interaction with the shell
    self.shell = '/bin/sh'
    # Preprocessing, compiling, and linking
    self.language     = []
    self.pushLanguage('C')
    return

  def addDefine(self, name, value, comment = ''):
    '''Designate that "name" should be defined to "value" in the configuration header'''
    self.defines[name] = value
    if comment: self.addHelp(name, comment)
    return

  def addSubstitution(self, name, value, comment = ''):
    '''Designate that "@name@" should be replaced by "value" in all files which experience substitution'''
    self.subst[name] = value
    if comment: self.addHelp(name, comment)
    return

  def addHelp(self, name, comment):
    '''Associate a help string with the variable "name"'''
    self.help[name] = comment
    return

  def getArgument(self, name, defaultValue = None, prefix = '', conversion = None, comment = ''):
    '''Define "self.name" to be the argument "name" if it was given, otherwise use "defaultValue"
    - "prefix" is just a string prefix for "name"
    - "conversion" is an optional conversion function for the string value
    '''
    if comment: self.addHelp(name, comment)
    argName = prefix+name
    value   = None
    if self.framework.argDB.has_key(argName):
      value = self.framework.argDB[argName]
    else:
      value = defaultValue
    if not value is None:
      if not conversion is None:
        setattr(self, name, conversion(value))
      else:
        setattr(self, name, value)
    return

  def getExecutable(self, name, path = '', getFullPath = 0, comment = ''):
    if not path: path = os.environ['PATH']
    for dir in path.split(':'):
      prog = os.path.join(dir, name)

      if os.path.isfile(prog) and os.access(prog, os.X_OK):
        if getFullPath:
          setattr(self, name, os.path.abspath(prog))
        else:
          setattr(self, name, name)
        self.addSubstitution(name.upper(), getattr(self, name), comment = comment)
        break
    return

  ###############################################
  # Preprocessor, Compiler, and Linker Operations
  def pushLanguage(self, language):
    self.language.append(language)
    return self.setLanguage(language)

  def popLanguage(self):
    self.language.pop()
    return self.setLanguage(self.language[-1])

  def setLanguage(self, language):
    self.language[-1] = language
    if language == 'C':
      self.compilerName = 'CC'
    elif language == 'C++':
      self.compilerName = 'CXX'
    elif language == 'F77':
      self.compilerName = 'FC'
    else:
      raise RuntimeError('Unknown language: '+language)

    if hasattr(self.framework, 'compilers'):
      self.compiler = getattr(self.framework.compilers, self.compilerName)
    else:
      self.compiler = self.framework.argDB[self.compilerName]

    if language == 'C':
      # Interaction with the preprocessor
      self.cpp        = self.framework.argDB['CPP']
      self.cppFlags   = self.framework.argDB['CPPFLAGS']
      self.cppCmd     = self.cpp+' '+self.cppFlags
      # Interaction with the compiler
      self.compilerDefines = 'confdefs.h'
      self.compilerSource  = 'conftest.c'
      self.compilerObj     = 'conftest.o'
      self.compilerFlags   = self.framework.argDB['CFLAGS']+' '+self.framework.argDB['CPPFLAGS']
      self.compilerCmd     = self.compiler+' -c -o '+self.compilerObj+' '+self.compilerFlags+' '+self.compilerSource
      # Interaction with the linker
      self.linker      = self.compiler
      self.linkerObj   = 'conftest'
      self.linkerFlags = self.framework.argDB['CFLAGS']+' '+self.framework.argDB['CPPFLAGS']+' '+self.framework.argDB['LDFLAGS']
      self.linkerCmd   = self.linker+' -o '+self.linkerObj+' '+self.linkerFlags+' conftest.o '+self.framework.argDB['LIBS']
    elif language == 'C++':
      # Interaction with the preprocessor
      self.cpp        = self.framework.argDB['CXXCPP']
      self.cppFlags   = self.framework.argDB['CPPFLAGS']
      self.cppCmd     = self.cpp+' '+self.cppFlags
      # Interaction with the compiler
      self.compilerDefines = 'confdefs.h'
      self.compilerSource  = 'conftest.cc'
      self.compilerObj     = 'conftest.o'
      self.compilerFlags   = self.framework.argDB['CXXFLAGS']+' '+self.framework.argDB['CPPFLAGS']
      self.compilerCmd     = self.compiler+' -c -o '+self.compilerObj+' '+self.compilerFlags+' '+self.compilerSource
      # Interaction with the linker
      self.linker      = self.compiler
      self.linkerObj   = 'conftest'
      self.linkerFlags = self.framework.argDB['CXXFLAGS']+' '+self.framework.argDB['CPPFLAGS']+' '+self.framework.argDB['LDFLAGS']
      self.linkerCmd   = self.linker+' -o '+self.linkerObj+' '+self.linkerFlags+' conftest.o '+self.framework.argDB['LIBS']
    elif language == 'F77':
      # Interaction with the preprocessor
      self.cpp        = self.framework.argDB['CXXCPP']
      self.cppFlags   = self.framework.argDB['CPPFLAGS']
      self.cppCmd     = self.cpp+' '+self.cppFlags
      # Interaction with the compiler
      self.compilerSource = 'conftest.f'
      self.compilerObj    = 'conftest.o'
      self.compilerFlags  = self.framework.argDB['FFLAGS']
      self.compilerCmd    = self.compiler+' -c -o '+self.compilerObj+' '+self.compilerFlags+' '+self.compilerSource
      # Interaction with the linker
      self.linker      = self.compiler
      self.linkerObj   = 'conftest'
      self.linkerFlags = self.framework.argDB['FFLAGS']+' '+self.framework.argDB['LDFLAGS']
      self.linkerCmd   = self.linker+' -o '+self.linkerObj+' '+self.linkerFlags+' conftest.o '+self.framework.argDB['LIBS']
    else:
      raise RuntimeError('Unknown language: '+language)
    return

  def getCode(self, includes, body = None):
    language = self.language[-1]
    if language == 'C' or language == 'C++':
      codeStr = '#include "confdefs.h"\n'+includes
      if not body is None:
        codeStr += '\nint main() {\n'+body+';\n  return 0;\n}\n'
    elif language == 'F77':
      if not body is None:
        codeStr = '      program main\n'+body+'\n      end'
    else:
      raise RuntimeError('Invalid language: '+language)
    return codeStr

  def outputPreprocess(self, codeStr):
    self.framework.outputHeader(self.compilerDefines)
    (input, output, err) = os.popen3(self.cppCmd)
    input.write(self.getCode(codeStr))
    input.close()
    out   = ''
    ready = select.select([output], [], [], 0.1)
    if len(ready[0]):
      out = ready[0][0].read()
    err.close()
    output.close()
    if os.path.isfile(self.compilerDefines): os.remove(self.compilerDefines)
    return out

  def checkPreprocess(self, codeStr):
    self.framework.outputHeader(self.compilerDefines)
    (input, output, err) = os.popen3(self.cppCmd)
    input.write(self.getCode(codeStr))
    input.close()
    out   = ''
    ready = select.select([err], [], [], 0.1)
    if len(ready[0]):
      # Log failure of preprocessor
      out = ready[0][0].read()
      if out: self.framework.log.write('ERR (preprocessor): '+out)
    err.close()
    output.close()
    if os.path.isfile(self.compilerDefines): os.remove(self.compilerDefines)
    return not len(out)

  def checkCompile(self, includes = '', body = '', cleanup = 1):
    self.framework.outputHeader(self.compilerDefines)
    f = file(self.compilerSource, 'w')
    f.write(self.getCode(includes, body))
    f.close()
    (input, output, err) = os.popen3(self.compilerCmd)
    input.close()
    out   = ''
    ready = select.select([err], [], [], 0.1)
    if len(ready[0]):
      # Log failure of compiler
      out = ready[0][0].read()
      if out:
        self.framework.log.write('ERR (compiler): '+out)
        self.framework.log.write('Source:\n'+self.getCode(includes, body))
    err.close()
    output.close()
    if os.path.isfile(self.compilerDefines): os.remove(self.compilerDefines)
    if os.path.isfile(self.compilerSource): os.remove(self.compilerSource)
    if cleanup and os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    return not len(out)

  def checkLink(self, includes, body, cleanup = 1):
    if not self.checkCompile(includes, body, cleanup = 0): return 0
    (input, output, err) = os.popen3(self.linkerCmd)
    input.write(self.getCode(includes, body))
    input.close()
    out   = ''
    ready = select.select([err], [], [], 0.1)
    if len(ready[0]):
      # Log failure of linker
      out = ready[0][0].read()
      if out: self.framework.log.write('ERR (linker): '+out)
    err.close()
    output.close()
    if os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    if cleanup and os.path.isfile(self.linkerObj): os.remove(self.linkerObj)
    return not len(out)

  def checkRun(self, includes, body):
    if not self.checkLink(includes, body, cleanup = 0): return 0
    success = 0
    (status, output) = commands.getstatusoutput('./'+self.linkerObj)
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
    newCode = re.sub('AC_FD_CC',  self.acCCFD,  newCode)
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
    ''' % (self.shell, self.acCCFD, self.acCCFD, self.acCCFD, self.framework.logName)

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

class Framework(Configure):
  def __init__(self, clArgs = None):
    self.argDB      = self.setupArgDB(clArgs)
    self.logName    = 'configure.log'
    self.log        = file(self.logName, 'w')
    Configure.__init__(self, self)
    self.children   = []
    self.substRE    = re.compile(r'@(?P<name>[^@]+)@')
    self.substFiles = {}
    self.header     = 'matt_config.h'
    return

  def setupArgDB(self, clArgs):
    return nargs.ArgDict('ArgDict', clArgs)

  def addSubstitutionFile(self, inName, outName = ''):
    '''Designate that file should experience substitution
      - If outName is given, inName --> outName
      - If inName == foo.in, foo.in --> foo
      - If inName == foo,    foo.in --> foo
    '''
    if outName:
      if inName == outName:
        raise RuntimeError('Input and output substitution files identical: '+inName)
    else:
      if inName[-3:] == '.in':
        root  = inName[-3:]
      else:
        root  = inName
      inName  = root+'.in'
      outName = root
    if not os.path.exists(inName):
      raise RuntimeError('Nonexistent substitution file: '+inName)
    self.substFiles[inName] = outName
    return

  def getPrefix(self, child):
    '''Get the default prefix for a given child Configure'''
    mod = child.__class__.__module__
    if not mod == '__main__':
      prefix = mod.replace('.', '_')
    else:
      prefix = ''
    return prefix

  def getHeaderPrefix(self, child):
    '''Get the prefix for variables in the configuration header for a given child'''
    if hasattr(child, 'headerPrefix'):
      prefix = child.headerPrefix
    else:
      prefix = self.getPrefix(child)
    return prefix

  def getSubstitutionPrefix(self, child):
    '''Get the prefix for variables during substitution for a given child'''
    if hasattr(child, 'substPrefix'):
      prefix = child.substPrefix
    else:
      prefix = self.getPrefix(child)
    return prefix

  def substituteName(self, match, prefix = None):
    '''Return the substitution value for a given name, or return "@name_UNKNOWN@"'''
    name = match.group('name')
    if self.subst.has_key(name):
      return self.subst[name]
    else:
      for child in self.children:
        if not hasattr(child, 'subst') or not isinstance(child.defines, dict): continue
        if prefix is None: prefix = self.getSubstitutionPrefix(child)
        if prefix:         prefix = prefix+'_'
        if prefix and name.startswith(prefix):
          childName = name.replace(prefix, '', 1)
        else:
          childName = name
        if child.subst.has_key(childName):
          return child.subst[childName]
    return '@'+name+'_UNKNOWN@'

  def substituteFile(self, inName, outName):
    '''Carry out substitution on the file "inName", creating "outName"'''
    inFile  = file(inName)
    outFile = file(outName, 'w')
    for line in inFile.xreadlines():
      outFile.write(self.substRE.sub(self.substituteName, line))
    outFile.close()
    inFile.close()

  def substitute(self):
    '''Preform all substitution'''
    for pair in self.substFiles.items():
      self.substituteFile(pair[0], pair[1])
    return

  def outputDefine(self, f, name, value = None, comment = ''):
    '''Define "name" to "value" in the configuration header'''
    name  = name.upper()
    guard = re.match(r'^(\w+)(\([\w,]+\))?', name).group(1)
    if comment:
      for line in comment.split('\n'):
        if line: f.write('/* '+line+' */\n')
    f.write('#ifndef '+guard+'\n')
    if value:
      f.write('#define '+name+' '+str(value)+'\n')
    else:
      f.write('/* #undef '+name+' */\n')
    f.write('#endif\n\n')

  def outputDefines(self, f, child, prefix = None):
    '''If the child contains a dictionary named "defines", the entries are output as defines in the config header.
    The prefix to each define is calculated as follows:
    - If the prefix argument is given, this is used, otherwise
    - If the child contains "headerPrefix", this is used, otherwise
    - If the module containing the child class is not "__main__", this is used, otherwise
    - No prefix is used
    If the child contains a dictinary name "help", then a help string will be added before the define
    '''
    if not hasattr(child, 'defines') or not isinstance(child.defines, dict): return
    if hasattr(child, 'help') and isinstance(child.help, dict):
      help = child.help
    else:
      help = {}
    if prefix is None: prefix = self.getHeaderPrefix(child)
    if prefix:         prefix = prefix+'_'
    for pair in child.defines.items():
      if help.has_key(pair[0]):
        self.outputDefine(f, prefix+pair[0], pair[1], help[pair[0]])
      else:
        self.outputDefine(f, prefix+pair[0], pair[1])
    return

  def outputHeader(self, name):
    '''Write the configuration header'''
    f = file(name, 'w')
    self.outputDefines(f, self)
    for child in self.children:
      self.outputDefines(f, child)
    f.close()
    return

  def checkTypes(self):
    import config.types
    self.types = config.types.Configure(self)
    self.children.append(self.types)
    return

  def checkHeaders(self, headers = []):
    import config.headers
    self.headers = config.headers.Configure(self, headers)
    self.children.append(self.headers)
    return

  def checkFunctions(self, functions = []):
    import config.functions
    self.functions = config.functions.Configure(self, functions)
    self.children.append(self.functions)
    return

  def checkCompilers(self):
    import config.compilers
    self.compilers = config.compilers.Configure(self)
    # It is important to check the compilers first
    self.children.insert(0, self.compilers)
    return

  def configure(self):
    '''Configure the system'''
    for child in self.children:
      child.configure()
    self.substitute()
    self.outputHeader(self.header)
    return

if __name__ == '__main__':
  framework = Framework(sys.argv[1:])
  conf      = PETSc.Configure.Configure(framework)
  framework.children.append(conf)
  framework.addSubstitutionFile('matt')
  framework.configure()
  #framework.compilers.configure()
  #conf.configure()
