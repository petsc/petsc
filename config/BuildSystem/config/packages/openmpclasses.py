import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions         = []
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.pthreadclasses = framework.require('config.packages.pthreadclasses',self)
    self.openmp         = framework.require('config.packages.openmp',self)
    self.deps           = [self.openmp, self.pthreadclasses]
    return

  def configureLibrary(self):
    if not self.openmp.found:
      raise RuntimeError('Needs support for OpenMP')
    # OpenMP threadprivate variables are not supported on all platforms (for e.g on MacOS).
    # Hence forcing to configure additionally with --with-pthreadclasses so that pthread
    # routines pthread_get/setspecific() can be used instead.
    if not self.checkCompile('#include <omp.h>\nint a;\n#pragma omp threadprivate(a)\n','') and not self.pthreadclasses.found:
      raise RuntimeError('OpenMP threadprivate variables not found. Configure additionally with --with-pthreadclasses=1')
    # register package since config.package.Package.configureLibrary(self) will not work since there is no library to find
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)
