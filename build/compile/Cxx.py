import build.processor

class Compiler (build.processor.Compiler):
  def __init__(self, sourceDB, usingCxx, compiler = 'g++', warningFlags = None, inputTag = 'cxx'):
    build.processor.Compiler.__init__(self, sourceDB, compiler, inputTag, updateType = 'deferred')
    self.usingCxx     = usingCxx
    self.warningFlags = warningFlags
    self.language     = 'Cxx'
    self.includeDirs.append('.')
    self.checkCompiler()
    return

  def __str__(self):
    return self.language+' compiler('+self.processor+') for '+str(self.inputTag)

  def checkCompiler(self):
    '''Checks the compatibility of the supplied compiler'''
    compiler = self.processor
    # Should be using isGNU from configure here
    if compiler.endswith('g++'):
      import commands
      # Make sure g++ is recent enough
      (status, output) = commands.getstatusoutput(compiler+' -dumpversion')
      if not status == 0:
        raise RuntimeError('The compiler you specified ('+compiler+') could not be run. Perhaps it is not in your path.')
      version = output.split('.')[0]
      if not version == '3':
        raise RuntimeError('The g++ you specified ('+compiler+') is version '+version+'; please install a g++ of at least version 3 or fix your path. Get gcc/g++ at http://gcc.gnu.com')
    return

  def getWarningFlags(self, source = None):
    '''Return a list of the compiler warning flags. The default is most of the GCC warnings.'''
    if self.warningFlags is None:
      return ['-Wall', '-Wundef', '-Wpointer-arith', '-Wbad-function-cast', '-Wcast-align', '-Wwrite-strings',
              '-Wconversion', '-Wsign-compare', '-Wstrict-prototypes', '-Wmissing-prototypes', '-Wmissing-declarations',
              '-Wmissing-noreturn', '-Wredundant-decls', '-Wnested-externs', '-Winline']
    return self.warningFlags

class MatlabCompiler (Compiler):
  def __init__(self, sourceDB, compiler = 'g++', warningFlags = None):
    Compiler.__init__(self, sourceDB, compiler, warningFlags)
    self.includeDirs.append(self.argDB['MATLAB_INCLUDE'])
    return

  def getLibraryName(self, source = None):
    '''Return the sahred library'''
    if not source: return None
    (dir, file) = os.path.split(source)
    (base, ext) = os.path.splitext(file)
    libraryName = os.path.join(dir, base+'.a')
    return libraryName
