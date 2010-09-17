import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
#    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/ams-dev.tar.gz']
#                      ssh://petsc@petsc.cs.iit.edu//hg/petsc/ams-dev 
#    Does not currently support automatic download and install
    self.functions = ['AMS_Memory_create']
    self.includes  = ['ams.h']
    self.liblist   = [['libamspub.a','libamsutilmt.a','libamsacc.a','libamsutil.a']]
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps       = [self.mpi]  
    return

  def configureLibrary(self):
    PETSc.package.NewPackage.configureLibrary(self)
    self.addDefine('AMS_DIR', '"'+os.path.dirname(self.include[0])+'"')
    self.addMakeMacro('AMS_DIR', '"'+os.path.dirname(self.include[0])+'"')    
