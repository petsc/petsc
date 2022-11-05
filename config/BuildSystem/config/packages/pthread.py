import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions         = ['pthread_create']
    self.includes          = ['pthread.h']
    self.liblist           = [['libpthread.a']]
    self.lookforbydefault  = 1
    self.pthread_barrier   = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.deps = []
    return

  def configureLibrary(self):
    ''' Checks for pthread_barrier_t '''
    config.package.Package.configureLibrary(self)
    if self.checkCompile('#include <pthread.h>', 'pthread_barrier_t *a;\n(void)a'):
      self.pthread_barrier = 1
