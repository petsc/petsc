import config.compile.processor
import config.compile.C

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
    self.requiredFlags = '-c'
    self.outputFlag    = '-o'
    self.child         = Preprocessor(argDB)
    return

class Linker(config.compile.processor.Processor):
  '''The Fortran linker'''
  def __init__(self, argDB):
    compiler        = Compiler(argDB)
    config.compile.processor.Processor.__init__(self, argDB, ['FC_LD', 'LD', compiler.name], 'LDFLAGS', '.o', '.a')
    self.outputFlag = '-o'
    if self.name == compiler.name:
      self.child    = compiler
    return

  def getExtraArguments(self):
    if not hasattr(self, '_extraArguments'):
      return self.argDB['LIBS']
    return self._extraArguments
  extraArguments = property(getExtraArguments, config.compile.processor.Processor.setExtraArguments, doc = 'Optional arguments for the end of the command')
