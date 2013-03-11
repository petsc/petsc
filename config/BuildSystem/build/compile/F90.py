import build.processor

class Compiler (build.processor.Compiler):
  def __init__(self, sourceDB, usingF90, compiler = None, warningFlags = None, inputTag = 'f90'):
    build.processor.Compiler.__init__(self, sourceDB, compiler, inputTag, updateType = 'deferred')
    self.usingF90     = usingF90
    self.warningFlags = warningFlags
    self.language     = 'F90'
    self.includeDirs.append('.')
    self.checkCompiler()
    return

  def __str__(self):
    return self.language+' compiler('+self.processor+') for '+str(self.inputTag)

  def checkCompiler(self):
    '''Checks the compatibility of the supplied compiler'''
    if self.processor is None:
      self.processor = self.argDB['F90']
    compiler = self.processor
    if not compiler == 'ifc':
      raise RuntimeError('I only know how to deal with Intel F90 right now. Shoot me.')
    return

  def getOptimizationFlags(self, source = None):
    if self.argDB['FFLAGS']:
      return [self.argDB['FFLAGS']]
    return []

  def getWarningFlags(self, source = None):
    '''Return a list of the compiler warning flags. The default is empty.'''
    if self.warningFlags is None:
      return []
    return self.warningFlags
