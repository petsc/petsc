import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions         = ['pthread_create']
    self.includes          = ['pthread.h']
    self.liblist           = [['libpthread.a']]
    self.complex           = 1   # 0 means cannot use complex
    self.lookforbydefault  = 1 
    self.double            = 0   # 1 means requires double precision 
    self.requires32bitint  = 0;  # 1 means that the package will not work with 64 bit integers
    self.worksonWindows    = 1  # 1 means that package can be used on Microsof Windows
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = []
    return

  def getSearchDirectories(self):
    ''' libpthread.a is in the usual place'''
    yield ''
    return

  def configureLibrary(self):
    ''' Checks for pthread.h and then checks if pthread_barrier_t'''
    PETSc.package.NewPackage.configureLibrary(self)
    if self.checkCompile('#include <pthread.h>', 'pthread_barrier_t *a;\n'):
      self.addDefine('HAVE_PTHREAD_BARRIER_T','1')
    if self.checkCompile('#include <sched.h>', 'cpu_set_t *a;\n'):
      self.addDefine('HAVE_SCHED_CPU_SET_T','1')
    
# sets PETSC_HAVE_PTHREAD but does NOT set PETSC_USE_PTHREAD; that is set only by particular packages that
# use pthreads
