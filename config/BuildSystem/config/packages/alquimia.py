import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'xsdk-0.2.0-rc1'
    self.download          = ['https://github.com/LBL-EESA/alquimia-dev/archive/'+self.gitcommit+'.tar.gz','git://https://github.com/LBL-EESA/alquimia-dev.git']
    self.functions         = []
    self.includes          = []
    self.hastests          = 1
    self.fc                = 1    # 1 means requires fortran
    self.cxx               = 1    # 1 means requires C++
    self.linkedbypetsc     = 0
    self.makerulename      = 'alquimia'    # make on the alquimia directory tries to build executables that will fail so force only building the libraries
    self.useddirectly      = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.mpi           = framework.require('config.packages.MPI', self)
    self.hdf5          = framework.require('config.packages.hdf5', self)
    self.pflotran      = framework.require('config.packages.pflotran', self)
    self.deps          = [self.mpi, self.hdf5, self.pflotran]
    return

  # the install is delayed until postProcess() since alquima install requires PETSc to be installed before alquima can be built.
  def Install(self):
    return self.installDir

  def configureLibrary(self):
    ''' Since alquimia cannot be built until after PETSc is compiled we need to just assume the downloaded library will work'''
    if 'with-alquimia' in self.framework.clArgDB:
      raise RuntimeError('Alquimia does not support --with-alquimia; only --download-alquimia')
    if 'with-alquimia-dir' in self.framework.clArgDB:
      raise RuntimeError('Alquimia does not support --with-alquimia-dir; only --download-alquimia')
    if 'with-alquimia-include' in self.framework.clArgDB:
      raise RuntimeError('Alquimia does not support --with-alquimia-include; only --download-alquimia')
    if 'with-alquimia-lib' in self.framework.clArgDB:
      raise RuntimeError('Alquimia does not support --with-alquimia-lib; only --download-alquimia')

    self.checkDownload()
    self.include = [os.path.join(self.installDir,'include')]
    self.lib     = [os.path.join(self.installDir,'lib','libalquimia.a')]
    self.found   = 1
    self.dlib    = self.lib
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')
    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=DEBUG')
      args.append('-DXSDK_ENABLE_DEBUG=YES')
    else:
      args.append('-DCMAKE_BUILD_TYPE=RELEASE')
      args.append('-DXSDK_ENABLE_DEBUG=NO')


    plibs = self.hdf5.lib
    if self.framework.argDB['prefix']:
       idir = os.path.join(self.getDefaultInstallDir(),'lib')
    else:
       idir = os.path.join(self.petscdir.dir,self.arch,'lib')
    if self.framework.argDB['with-single-library']:
      plibs = self.libraries.toStringNoDupes(['-L'+idir,' -lpetsc']+plibs)
    else:
      plibs = self.libraries.toStringNoDupes(['-L'+idir,'-lpetscts -lpetscsnes -lpetscksp -lpetscdm -lpetscmat -lpetscvec -lpetscsys']+plibs)

    args.append('-DTPL_PETSC_LDFLAGS="'+plibs+'"')
    args.append('-DTPL_PETSC_INCLUDE_DIRS="'+os.path.join(self.petscdir.dir,'include')+';'+';'.join(self.hdf5.include)+'"')

    args.append('-DXSDK_WITH_PFLOTRAN=ON')
    args.append('-DTPL_PFLOTRAN_LIBRARIES='+self.pflotran.lib[0])
    args.append('-DTPL_PFLOTRAN_INCLUDE_DIRS='+self.pflotran.include[0])
    return args

  def postProcess(self):
    # since we know that pflotran was built before alquimia we know that self.compilePETSc() has already run and installed PETSc
    #alquimia cmake requires PETSc environmental variables
    os.environ['PETSC_DIR']  = self.petscdir.dir
    os.environ['PETSC_ARCH'] = self.arch
    config.package.CMakePackage.Install(self)
    if not self.argDB['with-batch']:
      try:
        self.logPrintBox('Testing Alquimia; this may take several minutes')
        output,err,ret  = config.package.CMakePackage.executeShellCommand('cd '+os.path.join(self.packageDir,'petsc-build')+' && '+self.make.make+' test_install',timeout=60, log = self.log)
        output = output+err
        self.log.write(output)
        if output.find('Failure') > -1:
          raise RuntimeError('Error running make test on Alquimia: '+output)
      except RuntimeError as e:
        raise RuntimeError('Error running make test on Alquimia: '+str(e))


