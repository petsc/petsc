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
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = []
    return

  def getSearchDirectories(self):
    '''By default, do not search any particular directories'''
    return [os.path.join('/usr','local'),os.path.join('/opt','local')]
  
  def Install(self):
    raise RuntimeError('--download-valgrind not supported\n')

  def configure(self):
    '''By default we look for valgrind, but don't stop if it is not found'''
    self.consistencyChecks()
    if self.framework.argDB['with-'+self.package]:
      # If clanguage is c++, test external packages with the c++ compiler
      self.libraries.pushLanguage(self.defaultLanguage)
      try:
        self.executeTest(self.configureLibrary)
      except:
        pass
      self.libraries.popLanguage()
    else:
      self.executeTest(self.alternateConfigureLibrary)
    return
