import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions              = []
    self.includes               = []
    self.useddirectly           = 0
    self.linkedbypetsc          = 0
    self.builtafterpetsc        = 1
    self.PrefixWriteCheck       = 0
    return

  def setupHelp(self,help):
    import nargs
    help.addArgument('slepc4py', '-with-slepc4py=<bool>', nargs.ArgBool(None, False, 'Build SLEPc Python bindings (slepc4py)'))
    return

  def __str__(self):
    if self.found:
      s = 'slepc4py:\n'
      if hasattr(self,'pythonpath'):
        s += '  PYTHONPATH: '+self.pythonpath+'\n'
      return s
    return ''

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.python          = framework.require('config.packages.Python',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.cython          = framework.require('config.packages.Cython',self)
    self.petsc4py        = framework.require('config.packages.petsc4py',self)
    self.slepc           = framework.require('config.packages.SLEPc',self)
    self.deps            = [self.petsc4py,self.slepc]
    self.odeps           = [self.cython]
    return

  def getDir(self):
    return os.path.join(self.slepc.getDir(),'src','binding','slepc4py')

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if 'with-slepc4py' in self.argDB and self.argDB['with-slepc4py']:
      if not 'download-slepc' in self.argDB or not self.argDB['download-slepc']:
        raise RuntimeError('The option --with-slepc4py requires --download-slepc')

  def Install(self):
    import os
    installLibPath = os.path.join(self.installDir, 'lib')
    if self.setCompilers.isDarwin(self.log):
      apple = 'You may need to\n (csh/tcsh) setenv MACOSX_DEPLOYMENT_TARGET 10.X\n (sh/bash) MACOSX_DEPLOYMENT_TARGET=10.X; export MACOSX_DEPLOYMENT_TARGET\nbefore running make on PETSc'
    else:
      apple = ''
    self.logClearRemoveDirectory()
    self.logResetRemoveDirectory()
    archflags = ""
    if self.setCompilers.isDarwin(self.log):
      if self.setCompilers.isARM(self.log):
        archflags = "ARCHFLAGS=\'-arch arm64\' "
      elif self.types.sizes['void-p'] == 4:
        archflags = "ARCHFLAGS=\'-arch i386\' "
      else:
        archflags = "ARCHFLAGS=\'-arch x86_64\' "

    # Set PETSC_DIR/PETSC_ARCH to point at the dir with the PETSc installation:
    # if DESTDIR is non-empty, then PETSc has been installed into staging dir
    # if prefix has been specified at config time, path to PETSc includes that prefix
    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      newdir = 'SLEPC_DIR=${DESTDIR}' + os.path.abspath(os.path.expanduser(self.argDB['prefix'])) + ' PETSC_DIR=${DESTDIR}' + os.path.abspath(os.path.expanduser(self.argDB['prefix'])) + ' PETSC_ARCH= '
    else:
      newdir = ''

    # Pass to setup.py if given, otherwise setup.py will autodetect
    numpy_include = self.argDB.get('with-numpy-include')
    if numpy_include is not None:
      newdir += 'NUMPY_INCLUDE="'+numpy_include+'" '

    self.addDefine('SLEPC4PY_INSTALL_PATH','"'+os.path.join(self.installdir.dir,'lib')+'"')
    cflags = ''
    # by default, multiple flags are added by setup.py (-DNDEBUG -O3 -g), no matter the type of PETSc build
    # this is problematic with Intel compilers, which take extremely long to compile bindings when using -g
    # so we instead force no additional flags (other than the ones already used by PETSc, i.e., CFLAGS)
    # TODO FIXME: this observation was made with Intel(R) oneAPI DPC++/C++ Compiler 2025.1.0 (2025.1.0.20250317), but it may be fixed in a subsequent release
    if config.setCompilers.Configure.isIntel(self.getCompiler(), self.log):
      cflags = 'CFLAGS=\'\' '
    self.addPost(self.packageDir, ['${RM} -rf build',
                                   newdir + archflags + cflags + ' PYTHONPATH=${PETSCPYTHONPATH} SLEPC_DIR=' + self.slepc.installDir + ' ' + self.python.pyexe + ' setup.py build',
                                   'MPICC=${PCC} ' + newdir + archflags + ' PYTHONPATH=${PETSCPYTHONPATH} SLEPC_DIR=' + self.slepc.installDir + ' ' + self.python.pyexe +' setup.py install --install-lib=' + installLibPath + ' $(if $(DESTDIR),--root=\'$(DESTDIR)\')'])
    self.pythonpath = installLibPath
    self.addTest('.', 'PYTHONPATH=%s:${PETSCPYTHONPATH} PETSC_OPTIONS="%s" ${MPIEXEC} -n ${PETSC4PY_NP} %s %s --verbose' % (installLibPath, '${PETSC_OPTIONS} -check_pointer_intensity 0 -error_output_stdout -malloc_dump ${PETSC_TEST_OPTIONS}', self.python.pyexe, os.path.join(self.packageDir, 'test', 'runtests.py')))

    self.found = True
    self.python.path.add(installLibPath)
    self.logPrintBox('slepc4py examples are available at '+os.path.join(self.packageDir,'demo'))
    return self.installDir

  def configureLibrary(self):
    self.getInstallDir()
