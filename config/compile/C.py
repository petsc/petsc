import config.compile.processor

class Preprocessor(config.compile.processor.Processor):
  '''The C preprocessor'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'CPP', 'CPPFLAGS', '.cpp', '.c')
    return

class Compiler(config.compile.processor.Processor):
  '''The C compiler'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'CC', 'CFLAGS', '.c', '.o')
    self.requiredFlags[-1] = '-c'
    self.outputFlag        = '-o'
    self.flagsName.extend(Preprocessor(argDB).flagsName)
    return

  def getTarget(self, source):
    import os

    base, ext = os.path.splitext(source)
    return base+'.o'

class Linker(config.compile.processor.Processor):
  '''The C linker'''
  def __init__(self, argDB):
    compiler        = Compiler(argDB)
    config.compile.processor.Processor.__init__(self, argDB, ['CC_LD', 'LD', compiler.name], 'LDFLAGS', '.o', '.a')
    self.outputFlag = '-o'
    if self.name == compiler.name:
      self.flagsName.extend(compiler.flagsName)
    return

  def getExtraArguments(self):
    if not hasattr(self, '_extraArguments'):
      return self.argDB['LIBS']
    return self._extraArguments
  extraArguments = property(getExtraArguments, config.compile.processor.Processor.setExtraArguments, doc = 'Optional arguments for the end of the command')
