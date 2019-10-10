import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions         = ['pthread_create']
    self.includes          = ['pthread.h']
    self.liblist           = [['libpthread.a']]
    self.complex           = 1   # 0 means cannot use complex
    self.lookforbydefault  = 1
    self.pthread_barrier   = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.deps = []
    return

  def configureLibrary(self):
    ''' Checks for pthread_barrier_t, cpu_set_t, and sys/sysctl.h '''
    config.package.Package.configureLibrary(self)
    if self.checkCompile('#include <pthread.h>', 'pthread_barrier_t *a;\n'):
      self.addDefine('HAVE_PTHREAD_BARRIER_T','1')
      self.pthread_barrier = 1
    if self.checkCompile('#include <sched.h>', 'cpu_set_t *a;\n'):
      self.addDefine('HAVE_SCHED_CPU_SET_T','1')
    if self.checkPreprocess('#include <sys/sysctl.h>'):
      self.addDefine('HAVE_SYS_SYSCTL_H','1')

