import build.processor

class Compiler (build.processor.Compiler):
  def __init__(self, sourceDB, usingC, compiler = None, warningFlags = None, inputTag = 'c'):
    build.processor.Compiler.__init__(self, sourceDB, compiler, inputTag, updateType = 'deferred')
    self.usingC       = usingC
    self.warningFlags = warningFlags
    self.language     = 'C'
    self.includeDirs.append('.')
    self.checkCompiler()
    return

  def __str__(self):
    return self.language+' compiler('+self.processor+') for '+str(self.inputTag)

  def checkCompiler(self):
    '''Checks the compatibility of the supplied compiler'''
    if self.processor is None:
      self.processor = self.argDB['CC']
    return

  def getOptimizationFlags(self, source = None):
    if self.argDB['CFLAGS']:
      return [self.argDB['CFLAGS']]
    return []

  def getWarningFlags(self, source = None):
    '''Return a list of the compiler warning flags. The default is most of the GCC warnings.'''
    if self.warningFlags is None:
      return ['-Wall', '-Wundef', '-Wpointer-arith', '-Wbad-function-cast', '-Wcast-align', '-Wwrite-strings',
              '-Wconversion', '-Wsign-compare', '-Wstrict-prototypes', '-Wmissing-prototypes', '-Wmissing-declarations',
              '-Wmissing-noreturn', '-Wredundant-decls', '-Wnested-externs', '-Winline']
    return self.warningFlags
