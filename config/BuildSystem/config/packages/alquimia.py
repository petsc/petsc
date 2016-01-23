import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    #  temporarily use a fork of Alquimia with needed changes in it. Pull request already made to alquimia developers
    self.gitcommit         = 'master'
    self.download          = ['git://https://git@github.com:petsc/alquimia-dev.git']
    self.functions         = []
    self.includes          = []
    self.hastests          = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.petscdir       = framework.require('PETSc.options.petscdir', self.setCompilers)
    self.mpi   = framework.require('config.packages.MPI', self)
    self.hdf5  = framework.require('config.packages.hdf5', self)
    self.pflotran  = framework.require('config.packages.pflotran', self)
    self.deps  = [self.mpi, self.hdf5, self.pflotran]
    return

  # the install is delayed until postProcess() since Alquimia requires PETSc 
  def Install(self):
    return self.installDir

  def configureLibrary(self):
    ''' Just assume the downloaded library will work'''
    if self.framework.clArgDB.has_key('with-alquimia'):
      raise RuntimeError('Alquimia does not support --with-alquimia; only --download-alquimia')
    if self.framework.clArgDB.has_key('with-alquimia-dir'):
      raise RuntimeError('Alquimia does not support --with-alquimia-dir; only --download-alquimia')
    if self.framework.clArgDB.has_key('with-alquimia-include'):
      raise RuntimeError('Alquimia does not support --with-alquimia-include; only --download-alquimia')
    if self.framework.clArgDB.has_key('with-alquimia-lib'):
      raise RuntimeError('Alquimia does not support --with-alquimia-lib; only --download-alquimia')
    if self.framework.clArgDB.has_key('with-alquimia-shared'):
      raise RuntimeError('Alquimia does not support --with-alquimia-shared')

    self.checkDownload()
    self.include = [os.path.join(self.installDir,'include')]
    self.lib     = [os.path.join(self.installDir,'lib','libalquimia_c.a'),os.path.join(self.installDir,'lib','libalquimia_cutils.a'),os.path.join(self.installDir,'lib','libalquimia_fortran.a')]
    self.found   = 1
    self.dlib    = self.lib
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DXSDK_WITH_PFLOTRAN=ON')
    args.append('-DTPL_PFLOTRAN_LIBRARIES='+self.pflotran.lib[0])
    args.append('-DTPL_PFLOTRAN_INCLUDE_DIRS='+self.pflotran.include[0])

    # do not build with shared libraries because they require PETSc libraries be built first;
    rejects = ['-DBUILD_SHARED_LIBS=on']
    args = [arg for arg in args if not arg in rejects]
    args.append('-DBUILD_SHARED_LIBS=off')
    return args


  def postProcess(self):
    config.package.CMakePackage.Install(self)


