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
    help.addArgument('PETSc', '-with-petsc4py=<bool>', nargs.ArgBool(None, False, 'Build PETSc Python bindings (petsc4py)'))
    help.addArgument('PETSc', '-with-petsc4py-test-np=<np>',nargs.ArgInt(None, None, min=1, help='Number of processes to use for petsc4py tests'))
    help.addArgument('PETSc', '-with-numpy-include=<dir>', nargs.Arg(None, None, 'Path to numpy headers from numpy.get_include() (default: autodetect)'))
    return

  def __str__(self):
    if self.found:
      s = 'petsc4py:\n'
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
    return

  def getDir(self):
    return os.path.join('src','binding','petsc4py')

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
      newdir = 'PETSC_DIR=${DESTDIR}' + os.path.abspath(os.path.expanduser(self.argDB['prefix'])) + ' PETSC_ARCH= '
    else:
      newdir = ''

    # Pass to setup.py if given, otherwise setup.py will autodetect
    numpy_include = self.argDB.get('with-numpy-include')
    if numpy_include is not None:
      newdir += 'NUMPY_INCLUDE="'+numpy_include+'" '

    self.addDefine('HAVE_PETSC4PY',1)
    self.addDefine('PETSC4PY_INSTALL_PATH','"'+os.path.join(self.installdir.dir,'lib')+'"')
    self.addMakeMacro('PETSC4PY','yes')
    cflags = ''
    # by default, multiple flags are added by setup.py (-DNDEBUG -O3 -g), no matter the type of PETSc build
    # this is problematic with Intel compilers, which take extremely long to compile bindings when using -g
    # so we instead force no additional flags (other than the ones already used by PETSc, i.e., CFLAGS)
    # TODO FIXME: this observation was made with Intel(R) oneAPI DPC++/C++ Compiler 2025.1.0 (2025.1.0.20250317), but it may be fixed in a subsequent release
    if config.setCompilers.Configure.isIntel(self.getCompiler(), self.log):
      cflags = 'CFLAGS=\'\' '
    self.addPost(self.packageDir, ['${RM} -rf build',
                                   newdir + archflags + cflags + self.python.pyexe + ' setup.py build',
                                   'MPICC=${PCC} ' + newdir + archflags + self.python.pyexe +' setup.py install --install-lib=' + installLibPath + ' $(if $(DESTDIR),--root=\'$(DESTDIR)\')'])
    self.pythonpath = installLibPath
    np = self.make.make_test_np
    if self.mpi.usingMPIUni:
      np = 1
    # TODO: some tests currently have issues with np > 4, this should be fixed
    np = min(np,4)
    if 'with-petsc4py-test-np' in self.argDB and self.argDB['with-petsc4py-test-np']:
      np = self.argDB['with-petsc4py-test-np']
    self.addMakeMacro('PETSC4PY_NP',np)
    self.addTest('.', 'PYTHONPATH=%s:${PETSCPYTHONPATH} PETSC_OPTIONS="%s" ${MPIEXEC} -n ${PETSC4PY_NP} %s %s --verbose' % (installLibPath, '${PETSC_OPTIONS} -check_pointer_intensity 0 -error_output_stdout -malloc_dump ${PETSC_TEST_OPTIONS}', self.python.pyexe, os.path.join(self.packageDir, 'test', 'runtests.py')))
    self.found = True
    self.python.path.add(installLibPath)
    return self.installDir

  def configureLibrary(self):
    import sys
    if not self.sharedLibraries.useShared and not self.setCompilers.isCygwin(self.log):
      raise RuntimeError('petsc4py requires PETSc be built with shared libraries; rerun with --with-shared-libraries')
    if sys.version_info < (3, 6):
      raise RuntimeError('petsc4py requires Python 3.6 at least')
    chkpkgs = ['numpy']
    if sys.version_info >= (3, 12):
      chkpkgs.append('setuptools')
    npkgs  = []
    for pkg in chkpkgs:
      if not getattr(self.python,pkg): npkgs.append(pkg)
    if npkgs:
      raise RuntimeError('petsc4py requires Python with "%s" module(s) installed!\n'
                         'Please install using package managers - for ex: "apt" or "dnf" (on linux),\n'
                         'or with "pip" using: %s -m pip install %s' % (" ".join(npkgs), self.python.pyexe, " ".join(npkgs)))
    self.getInstallDir()

  def alternateConfigureLibrary(self):
    '''This is ugly but currently .gitlab-ci.yml is hardwired to use petsc4pytest'''
    self.addMakeRule('petsc4pytest','')

