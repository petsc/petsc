import config.compile.processor

class Preprocessor(config.compile.processor.Processor):
  '''The C++ preprocessor'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'CXXCPP', 'CPPFLAGS', '.cpp', '.cc')
    return

class Compiler(config.compile.processor.Processor):
  '''The C compiler'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'CXX', ['CXXFLAGS', 'CXX_CXXFLAGS'], '.cc', '.o')
    self.requiredFlags[-1] = '-c'
    self.outputFlag        = '-o'
    self.child             = Preprocessor(argDB)
    return

class Linker(config.compile.processor.Processor):
  '''The C linker'''
  def __init__(self, argDB):
    compiler        = Compiler(argDB)
    config.compile.processor.Processor.__init__(self, argDB, ['CXX_LD', 'LD', compiler.name], 'LDFLAGS', '.o', '.a')
    self.outputFlag = '-o'
    if self.name == compiler.name:
      self.child    = compiler
    return

  def getExtraArguments(self):
    if not hasattr(self, '_extraArguments'):
      return self.argDB['LIBS']
    return self._extraArguments
  extraArguments = property(getExtraArguments, config.compile.processor.Processor.setExtraArguments, doc = 'Optional arguments for the end of the command')
