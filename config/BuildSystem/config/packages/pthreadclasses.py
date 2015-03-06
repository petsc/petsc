import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions         = 0
    self.includes          = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.pthread = framework.require('config.packages.pthread',self)
    self.deps    = []
    return

  def configureLibrary(self):
    self.checkDependencies()
    if not self.pthread.found:
       raise RuntimeError('Pthreads not found, pthread classes needs pthreads to run')
    self.found = 1
    self.framework.packages.append(self)
    if self.checkCompile('__thread int a;\n',''):
      self.addDefine('PTHREAD_LOCAL','__thread')
    elif self.checkCompile('__declspec(thread) int i;\n',''):
      self.addDefine('PTHREAD_LOCAL','__declspec(thread)')
    return


