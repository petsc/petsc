import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'v1.0.2'
    self.download          = ['https://github.com/LBL-EESA/alquimia-dev/archive/'+self.gitcommit+'.tar.gz','git://https://github.com/LBL-EESA/alquimia-dev.git']
    self.functions         = []
    self.includes          = []
    self.hastests          = 1
    self.fc                = 1    # 1 means requires fortran
    self.cxx               = 1    # 1 means requires C++
    self.linkedbypetsc     = 0
    self.makerulename      = 'alquimia'    # make on the alquimia directory tries to build executables that will fail so force only building the libraries
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.installdir    = framework.require('PETSc.options.installDir',  self)
    self.petscdir      = framework.require('PETSc.options.petscdir', self.setCompilers)
    self.mpi           = framework.require('config.packages.MPI', self)
    self.hdf5          = framework.require('config.packages.hdf5', self)
    self.pflotran      = framework.require('config.packages.pflotran', self)
    self.deps          = [self.mpi, self.hdf5, self.pflotran]
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
       idir = os.path.join(self.installdir.dir,'lib')
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
    #alquimia cmake requires PETSc environmental variables
    os.environ['PETSC_DIR']  = self.petscdir.dir
    os.environ['PETSC_ARCH'] = self.arch
    config.package.CMakePackage.Install(self)
    if not self.argDB['with-batch']:
      try:
        self.logPrintBox('Testing Alquimia; this may take several minutes')
        output,err,ret  = config.package.CMakePackage.executeShellCommand('cd '+os.path.join(self.packageDir,'build')+' && '+self.make.make+' test_install',timeout=50, log = self.log)
        output = output+err
        self.log.write(output)
        if output.find('Failure') > -1:
          raise RuntimeError('Error running make test on Alquimia: '+output)
      except RuntimeError, e:
        raise RuntimeError('Error running make test on Alquimia: '+str(e))


