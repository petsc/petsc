import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = 0
    self.functions = []
    self.includes  = ['valgrind/valgrind.h']
    self.liblist   = ['']
    self.needsMath = 0
    self.complex   = 1
    self.required  = 1
    self.double    = 0
    self.requires32bitint = 0
    self.worksonWindows   = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = []
    return

  def getSearchDirectories(self):
    '''By default, do not search any particular directories'''
    yield ''
    yield os.path.join('/usr','local')
    yield os.path.join('/opt','local')
    return
  
  def Install(self):
    raise RuntimeError('--download-valgrind not supported\n')

  def configure(self):
    '''By default we look for valgrind, but do not stop if it is not found'''
    self.consistencyChecks()
    found = 0
    if self.framework.argDB['with-'+self.package]:
      if self.cxx:
        self.libraries.pushLanguage('C++')
      else:
        self.libraries.pushLanguage(self.defaultLanguage)
      try:
        self.executeTest(self.configureLibrary)
        oldFlags = self.compilers.CPPFLAGS
        self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
        if self.checkCompile('#include <valgrind/valgrind.h>', 'RUNNING_ON_VALGRIND;\n'):
          found = 1
        self.compilers.CPPFLAGS = oldFlags
      except:
        pass
      if not found and (self.setCompilers.isDarwin() or self.setCompilers.isLinux()):
        self.logPrintBox('It appears you do not have valgrind installed on your system.\n\
We HIGHLY recommend you install it from www.valgrind.org\n\
Or install valgrind-devel or equivalent using your package manager.\n\
Then rerun ./configure')
      self.libraries.popLanguage()
    else:
      self.executeTest(self.alternateConfigureLibrary)
    return
