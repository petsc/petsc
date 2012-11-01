from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download        = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/txpetscgpu-0.0.8.tar.gz']
    self.includes        = ['txpetscgpu_version.h']
    self.includedir      = ['include']
    self.forceLanguage   = 'CUDA'
    self.cxx             = 0
    self.archIndependent = 1
    self.worksonWindows  = 1
    self.downloadonWindows = 1
    self.complex         = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.thrust = framework.require('config.packages.thrust', self)
    self.cusp = framework.require('config.packages.cusp', self)
    self.deps   = [self.thrust, self.cusp]
    return

  def Install(self):
    import shutil
    import os
    self.framework.log.write('txpetscgpu directory = '+self.packageDir+' installDir '+self.installDir+'\n')
    srcdir = self.packageDir
    destdir = os.path.join(self.installDir, 'include', 'txpetscgpu')
    try:
      if os.path.isdir(destdir): shutil.rmtree(destdir)
      shutil.copytree(srcdir,destdir)
    except RuntimeError,e:
      raise RuntimeError('Error installing txpetscgpu include files: '+str(e))
# default and --download have different includedirs
    self.includedir = os.path.join(destdir, 'include')
    return self.installDir

  def getSearchDirectories(self):
    import os
    yield ''
    yield os.path.join('/usr','local','cuda')
    yield os.path.join('/usr','local','cuda','include')
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional tests needed by txpetscgpu'''
    if not self.cusp.found or not self.thrust.found:
      raise RuntimeError('PETSc TxPETScGPU support requires the CUSP and Thrust packages\nRerun configure using --with-cusp-dir and --with-thrust-dir')
    config.package.Package.configureLibrary(self)
    return

