import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions         = ['pthread_create']
    self.includes          = ['pthread.h']
    self.liblist           = [['libpthread.a']]
    self.complex           = 1
    self.lookforbydefault  = 1 
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = []
    return

  def getSearchDirectories(self):
    ''' libpthread.a is in the usual place'''
    return []

# sets PETSC_HAVE_PHTHREAD but does NOT set PETSC_USE_PTHREAD; that is set only by particular packages that
# use pthreads
