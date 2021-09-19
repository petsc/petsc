import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions              = []
    self.includes               = []
    self.skippackagewithoptions = 1
    self.useddirectly           = 0
    self.linkedbypetsc          = 0
    self.builtafterpetsc        = 1
    return

  def setupHelp(self,help):
    import nargs
    help.addArgument('PETSc', '-with-petsc4py=<bool>', nargs.ArgBool(None, False, 'Build PETSc Python bindings (petsc4py)'))
    help.addArgument('PETSc', '-with-petsc4py-test-np=<np>',nargs.ArgInt(None, None, min=1, help='Number of processes to use for petsc4py tests'))
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.python          = framework.require('config.packages.python',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
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
      if self.types.sizes['void-p'] == 4:
        archflags = "ARCHFLAGS=\'-arch i386\' "
      else:
        archflags = "ARCHFLAGS=\'-arch x86_64\' "

    # Set PETSC_DIR/PETSC_ARCH to point at the dir with the PETSc installation:
    # if DESTDIR is non-empty, then PETSc has been installed into staging dir
    # if prefix has been specified at config time, path to PETSc includes that prefix
    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      newdir = 'PETSC_DIR=${DESTDIR}'+os.path.abspath(os.path.expanduser(self.argDB['prefix'])) + \
              ' PETSC_ARCH= '
    else:
      newdir = ''

    newdir += 'MPICC=${PCC} '

    self.addDefine('HAVE_PETSC4PY',1)
    self.addDefine('PETSC4PY_INSTALL_PATH','"'+os.path.join(self.installdir.dir,'lib')+'"')
    self.addMakeMacro('PETSC4PY','yes')
    self.addMakeRule('petsc4pybuild','', \
                       ['@echo "*** Building petsc4py ***"',\
                          '@${RM} -f ${PETSC_ARCH}/lib/petsc/conf/petsc4py.errorflg',\
                          '@(cd '+self.packageDir+' && \\\n\
           '+newdir+archflags+self.python.pyexe+' setup.py build )  || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building petsc4py." && \\\n\
             echo "********************************************************************" && \\\n\
             touch ${PETSC_ARCH}/lib/petsc/conf/petsc4py.errorflg && \\\n\
             exit 1)'])
    self.addMakeRule('petsc4pyinstall','', \
                       ['@echo "*** Installing petsc4py ***"',\
                          '@(MPICC=${PCC} && export MPICC && cd '+self.packageDir+' && \\\n\
           '+newdir+archflags+self.python.pyexe+' setup.py install --install-lib='+installLibPath+' \\\n\
               $(if $(DESTDIR),--root=\'$(DESTDIR)\') ) || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building petsc4py." && \\\n\
             echo "********************************************************************" && \\\n\
             exit 1)',\
                          '@echo "====================================="',\
                          '@echo "To use petsc4py, add '+installLibPath+' to PYTHONPATH"',\
                          '@echo "====================================="'])

    np = self.make.make_test_np
    # TODO: some tests currently have issues with np > 4, this should be fixed
    np = min(np,4)
    if 'with-petsc4py-test-np' in self.argDB and self.argDB['with-petsc4py-test-np']:
      np = self.argDB['with-petsc4py-test-np']
    self.addMakeMacro('PETSC4PY_NP',np)
    self.addMakeRule('petsc4pytest', '',
        ['@echo "*** Testing petsc4py on ${PETSC4PY_NP} processes ***"',
         '@PYTHONPATH=%s:${PYTHONPATH} ${MPIEXEC} -n ${PETSC4PY_NP} %s %s --verbose' % \
             (installLibPath, self.python.pyexe, os.path.join(self.packageDir, 'test', 'runtests.py')),
         '@echo "====================================="'])

    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      self.addMakeRule('petsc4py-build','')
      # the build must be done at install time because PETSc shared libraries must be in final location before building petsc4py
      self.addMakeRule('petsc4py-install','petsc4pybuild petsc4pyinstall')
    else:
      self.addMakeRule('petsc4py-build','petsc4pybuild petsc4pyinstall')
      self.addMakeRule('petsc4py-install','')
    return self.installDir

  def configureLibrary(self):
    if not self.sharedLibraries.useShared and not self.setCompilers.isCygwin(self.log):
        raise RuntimeError('petsc4py requires PETSc be built with shared libraries; rerun with --with-shared-libraries')
    chkpkgs = ['cython','numpy']
    npkgs  = []
    for pkg in chkpkgs:
      if not getattr(self.python,pkg): npkgs.append(pkg)
    if npkgs:
      raise RuntimeError('PETSc4py requires Python with "%s" module(s) installed!\n'
                         'Please install using package managers - for ex: "apt" or "dnf" (on linux),\n'
                         'or with "pip" using: %s -m pip install %s' % (" ".join(npkgs), self.python.pyexe, " ".join(npkgs)))
    self.getInstallDir()

  def alternateConfigureLibrary(self):
    self.addMakeRule('petsc4py-build','')
    self.addMakeRule('petsc4py-install','')
    self.addMakeRule('petsc4pytest','')

