import os
import args
import config.compile.processor
import config.compile.C
import config.framework
import config.libraries

import config.setsOrdered as sets

class Preprocessor(config.compile.C.Preprocessor):
  '''The Fortran preprocessor, which now is just the C preprocessor'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'FPP', 'FPPFLAGS', '.FPP', '.F90')
    self.language        = 'FC'
    self.includeDirectories = sets.Set()
    return

class Compiler(config.compile.processor.Processor):
  '''The Fortran compiler'''
  def __init__(self, argDB, usePreprocessorFlags = True):
    config.compile.processor.Processor.__init__(self, argDB, 'FC', 'FFLAGS', '.F90', '.o')
    self.language           = 'FC'
    self.requiredFlags[-1]  = '-c'
    self.outputFlag         = '-o'
    self.includeDirectories = sets.Set()
    if usePreprocessorFlags:
      self.flagsName.extend(Preprocessor(argDB).flagsName)
    return

  def getTarget(self, source):
    base, ext = os.path.splitext(source)
    return base+'.o'

class Linker(config.compile.processor.Processor):
  '''The Fortran linker'''
  def __init__(self, argDB):
    self.compiler        = Compiler(argDB, usePreprocessorFlags = False)
    self.configLibraries = config.libraries.Configure(config.framework.Framework(clArgs = '', argDB = argDB, tmpDir = os.getcwd()))
    config.compile.processor.Processor.__init__(self, argDB, ['FC_LD', 'LD', self.compiler.name], ['LDFLAGS', 'FC_LINKER_FLAGS'], '.o', '.a')
    self.language   = 'FC'
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
        flagsName.extend(self.compiler.flagsName)
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
    import sys

    base, ext = os.path.splitext(source)
    if sys.platform[:3] == 'win' or sys.platform == 'cygwin':
      return base+'.exe'
    return base
