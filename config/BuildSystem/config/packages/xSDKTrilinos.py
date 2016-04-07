import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'master'
    self.download          = ['git://https://github.com/trilinos/xSDKTrilinos.git']
    self.downloaddirname   = 'xSDKTrilinos'
    self.includes          = []
    self.functions         = []
    self.cxx               = 1
    self.requirescxx11     = 1
    self.downloadonWindows = 0
    self.hastests          = 1
    self.linkedbypetsc     = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.petscdir       = framework.require('PETSc.options.petscdir', self.setCompilers)
    self.trilinos        = framework.require('config.packages.Trilinos',self)
    self.hypre           = framework.require('config.packages.hypre',self)
    #
    # also requires the ./configure option --with-cxx-dialect=C++11
    return

  # the install is delayed until postProcess() since xSDKTrilinos requires PETSc
  def Install(self):
    return self.installDir

  def configureLibrary(self):
    ''' Just assume the downloaded library will work'''
    if self.framework.clArgDB.has_key('with-xsdktrilinos'):
      raise RuntimeError('Xsdktrilinos does not support --with-xsdktrilinos; only --download-xsdktrilinos')
    if self.framework.clArgDB.has_key('with-xsdktrilinos-dir'):
      raise RuntimeError('Xsdktrilinos does not support --with-xsdktrilinos-dir; only --download-xsdktrilinos')
    if self.framework.clArgDB.has_key('with-xsdktrilinos-include'):
      raise RuntimeError('Xsdktrilinos does not support --with-xsdktrilinos-include; only --download-xsdktrilinos')
    if self.framework.clArgDB.has_key('with-xsdktrilinos-lib'):
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
    args.append('-DPETSC_LIBRARY_DIRS='+os.path.join(self.petscdir.dir,'lib'))
    args.append('-DPETSC_INCLUDE_DIRS='+os.path.join(self.petscdir.dir,'include'))

    args.append('-DxSDKTrilinos_ENABLE_TESTS=ON')
    return args

  def postProcess(self):
    self.compilePETSc()
    config.package.CMakePackage.Install(self)
    if not self.argDB['with-batch']:
      try:
        self.logPrintBox('Testing xSDKTrilinos; this may take several minutes')
        output,err,ret  = config.package.CMakePackage.executeShellCommand('cd '+os.path.join(self.packageDir,'build')+' && '+self.cmake.ctest,timeout=50, log = self.log)
        output = output+err
        self.log.write(output)
        if output.find('Failure') > -1:
          raise RuntimeError('Error running ctest on xSDKTrilinos: '+output)
      except RuntimeError, e:
        raise RuntimeError('Error running ctest on xSDKTrilinos: '+str(e))






