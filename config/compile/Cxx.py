import args
import config.compile.processor
import config.framework
import config.libraries

try:
  import sets
except ImportError:
  import config.setsBackport as sets

class Preprocessor(config.compile.processor.Processor):
  '''The C++ preprocessor'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'CXXCPP', 'CPPFLAGS', '.cpp', '.cc')
    return

class Compiler(config.compile.processor.Processor):
  '''The C++ compiler'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'CXX', ['CXXFLAGS', 'CXX_CXXFLAGS'], '.cc', '.o')
    self.requiredFlags[-1]  = '-c'
    self.outputFlag         = '-o'
    self.includeDirectories = sets.Set()
    self.flagsName.extend(Preprocessor(argDB).flagsName)
    return

  def getTarget(self, source):
    '''Return None for header files'''
    import os

    base, ext = os.path.splitext(source)
    if ext in ['.h', '.hh']:
      return None
    return base+'.o'

  def getCommand(self, sourceFiles, outputFile = None):
    '''If no outputFile is given, do not execute anything'''
    if outputFile is None:
      return 'true'
    return config.compile.processor.Processor.getCommand(self, sourceFiles, outputFile)

class Linker(config.compile.processor.Processor):
  '''The C++ linker'''
  def __init__(self, argDB):
    self.compiler        = Compiler(argDB)
    self.configLibraries = config.libraries.Configure(config.framework.Framework(clArgs = '', argDB = argDB))
    config.compile.processor.Processor.__init__(self, argDB, ['CXX_LD', 'LD', self.compiler.name], 'LDFLAGS', '.o', '.a')
    self.outputFlag = '-o'
    self.libraries  = sets.Set()
    return

  def copy(self, other):
    other.compiler = self.compiler
    other.configLibraries = self.configLibraries
    other.libraries = sets.Set(self.libraries)
    return

  def setArgDB(self, argDB):
    args.ArgumentProcessor.setArgDB(self, argDB)
    self.compiler.argDB                  = argDB
    self.configLibraries.argDB           = argDB
    self.configLibraries.framework.argDB = argDB
    return
  argDB = property(args.ArgumentProcessor.getArgDB, setArgDB, doc = 'The RDict argument database')

  def getFlags(self):
    '''Returns a string with the flags specified for running this processor.'''
    if not hasattr(self, '_flags'):
      flagsName = self.flagsName[:]
      if self.name == self.compiler.name:
        flagsName.append(self.compiler.flagsName[0])
      if hasattr(self, 'configCompilers'):
        flags = ' '.join([getattr(self.configCompilers, name) for name in flagsName])
      else:
        flags = ' '.join([self.argDB[name] for name in flagsName])
      return flags
    return self._flags
  flags = property(getFlags, config.compile.processor.Processor.setFlags, doc = 'The flags for the executable')

  def getExtraArguments(self):
    if not hasattr(self, '_extraArguments'):
      return self.configCompilers.LIBS
    return self._extraArguments
  extraArguments = property(getExtraArguments, config.compile.processor.Processor.setExtraArguments, doc = 'Optional arguments for the end of the command')

  def getTarget(self, source, shared):
    import os
    import sys

    base, ext = os.path.splitext(source)
    if sys.platform[:3] == 'win' or sys.platform == 'cygwin':
      return base+'.exe'
    return base
