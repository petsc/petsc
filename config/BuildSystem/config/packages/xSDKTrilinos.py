import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = '0ddbf6e' # master
    self.download          = ['https://github.com/trilinos/xSDKTrilinos/archive/'+self.gitcommit+'.tar.gz','git://https://github.com/trilinos/xSDKTrilinos.git']
    self.downloaddirnames  = ['xSDKTrilinos']
    self.includes          = []
    self.functions         = []
    self.buildLanguages    = ['Cxx']
    self.downloadonWindows = 0
    self.hastests          = 1
    self.linkedbypetsc     = 0
    self.useddirectly      = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.trilinos = framework.require('config.packages.Trilinos',self)
    self.hypre    = framework.require('config.packages.hypre',self)
    self.x        = framework.require('config.packages.X',self)
    self.ssl      = framework.require('config.packages.ssl',self)
    self.triangle = framework.require('config.packages.Triangle',self)
    self.exodusii = framework.require('config.packages.exodusii',self)
    self.flibs    = framework.require('config.packages.flibs',self)
    self.cxxlibs  = framework.require('config.packages.cxxlibs',self)
    self.mathlib  = framework.require('config.packages.mathlib',self)
    return

  # the install is delayed until postProcess() since xSDKTrilinos install requires PETSc to be installed before xSDKTrilinos can be built
  def Install(self):
    return self.installDir

  def configureLibrary(self):
    ''' Since xSDKTrilinos cannot be built until after PETSc is compiled we need to just assume the downloaded library will work'''
    if 'with-xsdktrilinos' in self.framework.clArgDB:
      raise RuntimeError('Xsdktrilinos does not support --with-xsdktrilinos; only --download-xsdktrilinos')
    if 'with-xsdktrilinos-dir' in self.framework.clArgDB:
      raise RuntimeError('Xsdktrilinos does not support --with-xsdktrilinos-dir; only --download-xsdktrilinos')
    if 'with-xsdktrilinos-include' in self.framework.clArgDB:
      raise RuntimeError('Xsdktrilinos does not support --with-xsdktrilinos-include; only --download-xsdktrilinos')
    if 'with-xsdktrilinos-lib' in self.framework.clArgDB:
      raise RuntimeError('Xsdktrilinos does not support --with-xsdktrilinos-lib; only --download-xsdktrilinos')

    self.checkDownload()
    self.include = [os.path.join(self.installDir,'include')]
    self.lib     = [os.path.join(self.installDir,'lib','libxsdktrilinos.a')]
    self.found   = 1
    self.dlib    = self.lib
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')
    args.append('-DTRILINOS_INSTALL_DIR='+os.path.dirname(self.trilinos.include[0]))
    args.append('-DTrilinos_INSTALL_DIR='+os.path.dirname(self.trilinos.include[0]))
    if self.hypre.found:
      args.append('-DTPL_ENABLE_HYPRE=ON')
      args.append('-DTPL_HYPRE_LIBRARIES="'+self.libraries.toStringNoDupes(self.hypre.lib)+'"')
      args.append('-DTPL_HYPRE_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.hypre.include)[2:])

    args.append('-DTPL_ENABLE_PETSC=ON')
    # These are packages that PETSc may be using that Trilinos is not be using
    plibs = self.exodusii.dlib+self.triangle.lib+self.ssl.lib+self.x.lib

    if not hasattr(self.compilers, 'FC'):
      args.append('-DxSDKTrilinos_ENABLE_Fortran=OFF')

    if self.framework.argDB['prefix']:
       idir = os.path.join(self.getDefaultInstallDir(),'lib')
    else:
       idir = os.path.join(self.petscdir.dir,self.getArch(),'lib')
    if self.framework.argDB['with-single-library']:
      plibs = self.libraries.toStringNoDupes(['-L'+idir,' -lpetsc']+plibs)
    else:
      plibs = self.libraries.toStringNoDupes(['-L'+idir,'-lpetscts -lpetscsnes -lpetscksp -lpetscdm -lpetscmat -lpetscvec -lpetscsys']+plibs)

    args.append('-DTPL_PETSC_LIBRARIES="'+plibs+'"')
    args.append('-DTPL_PETSC_INCLUDE_DIRS='+os.path.join(self.petscdir.dir,'include'))

    if self.compilerFlags.debugging:
      args.append('-DxSDKTrilinos_ENABLE_DEBUG=YES')
    else:
      args.append('-DxSDKTrilinos_ENABLE_DEBUG=NO')

    args.append('-DxSDKTrilinos_EXTRA_LINK_FLAGS="'+self.libraries.toStringNoDupes(self.flibs.lib+self.cxxlibs.lib+self.mathlib.lib)+' '+self.compilers.LIBS+'"')
    args.append('-DxSDKTrilinos_ENABLE_TESTS=ON')
    return args

  def postProcess(self):
    self.compilePETSc()
    config.package.CMakePackage.Install(self)
    if not self.argDB['with-batch']:
      try:
        self.logPrintBox('Testing xSDKTrilinos; this may take several minutes')
        output,err,ret  = config.package.CMakePackage.executeShellCommand('cd '+os.path.join(self.packageDir,'petsc-build')+' && '+self.cmake.ctest,timeout=60, log = self.log)
        output = output+err
        self.log.write(output)
        if output.find('Failure') > -1:
          raise RuntimeError('Error running ctest on xSDKTrilinos: '+output)
      except RuntimeError as e:
        raise RuntimeError('Error running ctest on xSDKTrilinos: '+str(e))
    else:
      self.logClear()
      self.logPrintDivider( debugSection = 'screen')
      self.logPrint('Since this is a batch system xSDKTrilinos cannot run tests directly. To run a short test suite.', debugSection = 'screen')
      self.logPrint('   Obtain an interactive session with your batch system', debugSection = 'screen')
      self.logPrint('   cd to '+os.path.join(self.packageDir,'petsc-build'), debugSection = 'screen')
      self.logPrint('   ctest', debugSection = 'screen')
      linewidth = self.linewidth
      self.linewidth = -1
      if self.hypre.found:
        self.logPrint(os.path.join(os.getcwd(),self.packageDir,'petsc-build','hypre','test','xSDKTrilinos_HypreTest.exe'), debugSection = 'screen')
      self.logPrint(os.path.join(os.getcwd(),self.packageDir,'petsc-build','petsc','test','xSDKTrilinos_PETScAIJMatrix.exe'), debugSection = 'screen')
      self.linewidth = linewidth
      self.logPrintDivider( debugSection = 'screen')
      self.logPrint('', debugSection = 'screen')






