#!/usr/bin/env python
import PETSc
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
    self.argSubst  = {}
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
    self.setupPreprocessor()
    self.setupCCompiler()
    self.setupCXXCompiler()
    self.setupF77Compiler()
    self.setupLinker()
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

  def addArgumentSubstitution(self, name, arg, comment = ''):
    '''Designate that "@name@" should be replaced by argDB["arg"] in all files which experience substitution'''
    self.argSubst[name] = arg
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
      name = name.replace('-', '_')
      if not conversion is None:
        setattr(self, name, conversion(value))
      else:
        setattr(self, name, value)
    return

  def getExecutable(self, name, path = '', getFullPath = 0, comment = '', resultName = ''):
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
        self.addSubstitution(resultName.upper(), getattr(self, resultName), comment = comment)
        found = 1
        self.framework.log.write('found\n')
        break
      self.framework.log.write('not found\n')
    return found

  def getExecutables(self, names, path = '', getFullPath = 0, comment = '', resultName = ''):
    for name in names:
      if self.getExecutable(name, path, getFullPath, comment, resultName):
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

  def setupCCompiler(self):
    if not self.framework.argDB.has_key('CC'):
      if not self.getExecutables(['gcc', 'cc', 'xlC', 'xlc', 'pgcc'], resultName = 'CC'):
        raise RuntimeError('Could not find a C compiler. Please set with the option -CC')
      self.framework.argDB['CC'] = self.CC
    if not self.framework.argDB.has_key('CFLAGS'):
      self.framework.argDB['CFLAGS'] = '-g -Wall'
    return self.framework.argDB['CC']

  def setupCXXCompiler(self):
    if not self.framework.argDB.has_key('CXX'):
      if not self.getExecutables(['g++', 'c++', 'CC', 'xlC', 'pgCC', 'cxx', 'cc++', 'cl'], resultName = 'CXX'):
        raise RuntimeError('Could not find a C++ compiler. Please set with the option -CXX')
      self.framework.argDB['CXX'] = self.CXX
    if not self.framework.argDB.has_key('CXXFLAGS'):
      self.framework.argDB['CXXFLAGS'] = '-g -Wall'
    return self.framework.argDB['CXX']

  def setupF77Compiler(self):
    if not self.framework.argDB.has_key('FC'):
      if not self.getExecutables(['g77', 'f77', 'pgf77'], resultName = 'FC'):
        raise RuntimeError('Could not find a Fortran 77 compiler. Please set with the option -FC')
      self.framework.argDB['FC'] = self.FC
    if not self.framework.argDB.has_key('FFLAGS'):
      self.framework.argDB['FFLAGS'] = '-g'
    return self.framework.argDB['FC']

  def setupPreprocessor(self):
    if not self.framework.argDB.has_key('CPP'):
      self.framework.argDB['CPP'] = self.setupCCompiler()+' -E'
    if not self.framework.argDB.has_key('CXXCPP'):
      self.framework.argDB['CXXCPP'] = self.setupCXXCompiler()+' -E'
    if not self.framework.argDB.has_key('CPPFLAGS'):
      self.framework.argDB['CPPFLAGS'] = ''
    return self.framework.argDB['CPP']

  def setupLinker(self):
    if not self.framework.argDB.has_key('LDFLAGS'):
      self.framework.argDB['LDFLAGS'] = ''
    return

  def getCompiler(self):
    language = self.language[-1]
    if language == 'C':
      self.compilerName   = 'CC'
      self.compilerSource = 'conftest.c'
      self.compilerObj    = 'conftest.o'
    elif language == 'C++':
      self.compilerName   = 'CXX'
      self.compilerSource = 'conftest.cc'
      self.compilerObj    = 'conftest.o'
    elif language == 'F77':
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

  def outputPreprocess(self, codeStr):
    command = self.getCppCmd()
    self.framework.outputHeader(self.compilerDefines)
    f = file(self.compilerSource, 'w')
    f.write(self.getCode(codeStr))
    f.close()
    self.framework.log.write('Executing: '+command+'\n')
    (input, output, err) = os.popen3(command)
    input.close()
    out   = ''
    ready = select.select([output], [], [], 0.1)
    if len(ready[0]):
      out = ready[0][0].read()
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
    (input, output, err) = os.popen3(command)
    input.close()
    out   = ''
    ready = select.select([err], [], [], 0.1)
    if len(ready[0]):
      # Log failure of preprocessor
      out = ready[0][0].read()
      if out:
        self.framework.log.write('ERR (preprocessor): '+out)
        self.framework.log.write('Source:\n'+self.getCode(codeStr))
    err.close()
    output.close()
    if os.path.isfile(self.compilerDefines): os.remove(self.compilerDefines)
    if os.path.isfile(self.compilerSource): os.remove(self.compilerSource)
    return not len(out)

  def outputCompile(self, includes = '', body = '', cleanup = 1):
    command = self.getCompilerCmd()
    self.framework.outputHeader(self.compilerDefines)
    f = file(self.compilerSource, 'w')
    f.write(self.getCode(includes, body))
    f.close()
    self.framework.log.write('Executing: '+command+'\n')
    (input, output, err) = os.popen3(command)
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
    return out

  def checkCompile(self, includes = '', body = '', cleanup = 1):
    return not len(self.outputCompile(includes, body, cleanup))

  def outputLink(self, includes, body, cleanup = 1):
    out = self.outputCompile(includes, body, cleanup = 0)
    if len(out): return out
    self.framework.log.write('Executing: '+self.getLinkerCmd()+'\n')
    (input, output, err) = os.popen3(self.getLinkerCmd())
    input.write(self.getCode(includes, body))
    input.close()
    out   = ''
    ready = select.select([err], [], [], 0.1)
    if len(ready[0]):
      # Log failure of linker
      out = ready[0][0].read()
      if out:
        self.framework.log.write('ERR (linker): '+out)
        self.framework.log.write(' in '+self.getLinkerCmd()+'\n')
    err.close()
    output.close()
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

class Help:
  def __init__(self, framework):
    self.framework = framework
    self.options   = {}
    self.sections  = []
    return

  def setTitle(self, title):
    self.title = title

  def addOption(self, section, name, comment):
    if self.options.has_key(section):
      if self.options[section].has_key(name):
        raise RuntimeError('Duplicate configure option '+name+' in section '+section)
      self.options[section][name] = comment
    else:
      self.sections.append(section)
      self.options[section] = {name: comment}
    return

  def output(self):
    print self.title
    for i in range(len(self.title)): sys.stdout.write('-')
    print
    nameLen = 1
    for section in self.sections:
      nameLen = max([nameLen, max(map(len, self.options[section].keys()))+1])
    for section in self.sections:
      print section+':'
      format  = '  -%-'+str(nameLen)+'s: %s'
      for item in self.options[section].items():
        print format % item
    return

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
    self.setFromOptions()
    return

  def setupArgDB(self, clArgs):
    return nargs.ArgDict('ArgDict', clArgs)

  def setFromOptions(self):
    if not self.argDB.has_key('configModules'):
      self.argDB['configModules'] = ['PETSc.Configure']
    if not isinstance(self.argDB['configModules'], list):
      self.argDB['configModules'] = [self.argDB['configModules']]
    for moduleName in self.argDB['configModules']:
      self.children.append(__import__(moduleName, globals(), locals(), ['Configure']).Configure(self))

  def require(self, moduleName, depChild = None, keywordArgs = {}):
    type   = __import__(moduleName, globals(), locals(), ['Configure']).Configure
    config = None
    for child in self.children:
      if isinstance(child, type):
        config = child
    if not config:
      config = apply(type, [self], keywordArgs)
      self.children.append(config)
    if depChild in self.children and self.children.index(config) > self.children.index(depChild):
      self.children.remove(config)
      self.children.insert(self.children.index(depChild), config)
    return config
        
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
    elif self.argSubst.has_key(name):
      return self.argDB[self.argSubst[name]]
    else:
      for child in self.children:
        if not hasattr(child, 'subst') or not isinstance(child.defines, dict):
          continue
        if prefix is None:
          substPrefix = self.getSubstitutionPrefix(child)
        else:
          substPrefix = prefix
        if substPrefix:
          substPrefix = substPrefix+'_'
          if name.startswith(substPrefix):
            childName = name.replace(substPrefix, '', 1)
          else:
            continue
        else:
          childName = name
        if child.subst.has_key(childName):
          return child.subst[childName]
        elif child.argSubst.has_key(childName):
          return self.argDB[child.argSubst[childName]]
    return '@'+name+'_UNKNOWN@'

  def substituteFile(self, inName, outName):
    '''Carry out substitution on the file "inName", creating "outName"'''
    inFile  = file(inName)
    if not os.path.exists(os.path.dirname(outName)):
      os.makedirs(os.path.dirname(outName))
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

  def dumpSubstitutions(self):
    for pair in self.subst.items():
      print pair[0]+'  --->  '+pair[1]
    for pair in self.argSubst.items():
      print pair[0]+'  --->  '+self.argDB[pair[1]]
    for child in self.children:
      if not hasattr(child, 'subst') or not isinstance(child.defines, dict): continue
      substPrefix = self.getSubstitutionPrefix(child)
      for pair in child.subst.items():
        if substPrefix:
          print substPrefix+'_'+pair[0]+'  --->  '+pair[1]
        else:
          print pair[0]+'  --->  '+pair[1]
      for pair in child.argSubst.items():
        if substPrefix:
          print substPrefix+'_'+pair[0]+'  --->  '+self.argDB[pair[1]]
        else:
          print pair[0]+'  --->  '+self.argDB[pair[1]]
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
    guard = 'INCLUDED_'+os.path.basename(name).upper().replace('.', '_')
    f.write('#if !defined('+guard+')\n')
    f.write('#define '+guard+'\n\n')
    self.outputDefines(f, self)
    for child in self.children:
      self.outputDefines(f, child)
    f.write('#endif /* '+guard+' */\n')
    f.close()
    return

  def configureHelp(self, help):
    help.addOption('Framework', 'configModules', 'A list of Python modules with a Configure class')
    return

  def configureClear(self):
    del self.argDB['CC']
    del self.argDB['CFLAGS']
    del self.argDB['CXX']
    del self.argDB['CXXFLAGS']
    del self.argDB['FC']
    del self.argDB['FFLAGS']
    del self.argDB['CPP']
    del self.argDB['CXXCPP']
    del self.argDB['CPPFLAGS']
    del self.argDB['LDFLAGS']
    del self.argDB['LIBS']
    del self.argDB['configModules']
    del self.argDB['clear']
    return

  def configure(self):
    '''Configure the system'''
    if self.argDB.has_key('clear') and int(self.argDB['clear']):
      self.configureClear()
      return
    if self.argDB.has_key('help') and int(self.argDB['help']):
      help = Help(self)
      help.setTitle('Python Configure Help')
      self.configureHelp(help)
      for child in self.children:
        if hasattr(child, 'configureHelp'): child.configureHelp(help)
      help.output()
      del self.argDB['help']
      return
    for child in self.children:
      print 'Configuring '+child.__module__
      child.configure()
    self.substitute()
    self.outputHeader(self.header)
    return

if __name__ == '__main__':
  framework = Framework(sys.argv[1:])
  framework.argDB['CPPFLAGS'] = ''
  framework.argDB['LIBS'] = ''
  framework.configure()
  framework.dumpSubstitutions()
