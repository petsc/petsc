import config.compile.processor
import config.compile.C
import config.framework
import config.libraries

class Preprocessor(config.compile.C.Preprocessor):
  '''The Fortran preprocessor, which now is just the C preprocessor'''
  def __init__(self, argDB):
    config.compile.C.Preprocessor.__init__(self, argDB)
    self.targetExtension = '.F'
    return

class Compiler(config.compile.processor.Processor):
  '''The Fortran compiler'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'FC', 'FFLAGS', '.F', '.o')
    self.requiredFlags[-1] = '-c'
    self.outputFlag        = '-o'
    self.flagsName.extend(Preprocessor(argDB).flagsName)
    return

  def getTarget(self, source):
    import os

    base, ext = os.path.splitext(source)
    return base+'.o'

class Linker(config.compile.processor.Processor):
  '''The Fortran linker'''
  def __init__(self, argDB):
    compiler        = Compiler(argDB)
    config.compile.processor.Processor.__init__(self, argDB, ['FC_LD', 'LD', compiler.name], 'LDFLAGS', '.o', '.a')
    self.outputFlag = '-o'
    self.libraries  = []
    if self.name == compiler.name:
      self.flagsName.extend(compiler.flagsName)
    self.configLibrary = config.libraries.Configure(config.framework.Framework('', self.argDB))
    return

  def getExtraArguments(self):
    if not hasattr(self, '_extraArguments'):
      return self.argDB['LIBS']
    return self._extraArguments
  extraArguments = property(getExtraArguments, config.compile.processor.Processor.setExtraArguments, doc = 'Optional arguments for the end of the command')

  def getTarget(self, source, shared):
    import os
    import sys

    base, ext = os.path.splitext(source)
    if shared:
      return base+'.so'
    if sys.platform[:3] == 'win' or sys.platform == 'cygwin':
      return base+'.exe'
    return base
