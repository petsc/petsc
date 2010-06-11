import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions = 0
    self.includes  = 0
    self.liblist   = 0
    return

  def configureLibrary(self):
    self.addDefine('HAVE_JAVA','1')
    if self.framework.argDB.has_key('java'):
      self.addMakeMacro('JAVA',self.framework.argDB['java'])
    else:
      self.getExecutable('java',   getFullPath = 1)
    if self.framework.argDB.has_key('javac'):
      self.addMakeMacro('JAVAC',self.framework.argDB['javac'])
    else:
      self.getExecutable('javac',  getFullPath = 1)        
