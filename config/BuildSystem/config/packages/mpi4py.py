import config.package
import os
import script

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download          = ['https://github.com/mpi4py/mpi4py/releases/download/4.1.0/mpi4py-4.1.0.tar.gz']
    self.functions         = []
    self.includes          = []
    self.useddirectly      = 0
    self.pythonpath        = ''
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.python          = framework.require('config.packages.Python',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.deps            = [self.mpi]
    return

  def __str__(self):
    if self.found:
      s = 'mpi4py:\n'
      if hasattr(self,'pythonpath'):
        s += '  PYTHONPATH: '+self.pythonpath+'\n'
      return s
    return ''

  def Install(self):
    installLibPath  = os.path.join(self.installDir, 'lib')
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

    self.framework.pushLanguage('C')
    self.logPrintBox('Building mpi4py, this may take several minutes')
    cleancmd = 'MPICC='+self.framework.getCompiler()+'  '+archflags+self.python.pyexe+' setup.py clean --all  2>&1'
    output,err,ret  = config.base.Configure.executeShellCommand(cleancmd, cwd=self.packageDir,  checkCommand=script.Script.passCheckCommand, timeout=100, log=self.log)
    if ret: raise RuntimeError('Error cleaning mpi4py. Check configure.log')

    cflags = ''
    # by default, multiple flags are added by setup.py (-DNDEBUG -O3 -g), no matter the type of PETSc build
    # this is problematic with Intel compilers, which take extremely long to compile bindings when using -g
    # so we instead force no additional flags (other than the ones already used by PETSc, i.e., CFLAGS)
    # TODO FIXME: this observation was made with Intel(R) oneAPI DPC++/C++ Compiler 2025.1.0 (2025.1.0.20250317), but it may be fixed in a subsequent release
    if config.setCompilers.Configure.isIntel(self.getCompiler(), self.log):
      cflags = 'CFLAGS=\''+self.getCompilerFlags()+'\' '
    buildcmd = 'MPICC='+self.framework.getCompiler()+'  '+archflags+cflags+self.python.pyexe+' setup.py build 2>&1'
    output,err,ret  = config.base.Configure.executeShellCommand(buildcmd, cwd=self.packageDir, checkCommand=script.Script.passCheckCommand,timeout=100, log=self.log)
    if ret: raise RuntimeError('Error building mpi4py. Check configure.log')

    self.logPrintBox('Installing mpi4py')
    installcmd = 'MPICC='+self.framework.getCompiler()+' '+self.python.pyexe+' setup.py install --install-lib='+installLibPath+' 2>&1'
    output,err,ret  = config.base.Configure.executeShellCommand(installcmd, cwd=self.packageDir,checkCommand=script.Script.passCheckCommand, timeout=100, log=self.log)
    if ret: raise RuntimeError('Error installing mpi4py. Check configure.log')
    self.framework.popLanguage()
    return self.installDir

  def configureLibrary(self):
    self.checkDownload()
    if not self.sharedLibraries.useShared:
        raise RuntimeError('mpi4py requires PETSc be built with shared libraries; rerun with --with-shared-libraries')
    if not getattr(self.python,'numpy'):
        raise RuntimeError('mpi4py, in the context of PETSc,requires Python with numpy module installed.\n'
                           'Please install using package managers - for ex: "apt" or "dnf" (on linux),\n'
                           'or  using: %s -m pip install %s' % (self.python.pyexe, 'numpy'))
    if self.argDB.get('with-mpi4py-dir'):
      self.directory = self.argDB['with-mpi4py-dir']
    elif self.argDB.get('download-mpi4py'):
      self.directory = os.path.join(self.installDir)

    elif self.argDB.get('with-mpi4py'):
      try:
        import mpi4py
      except:
        raise RuntimeError('mpi4py not found in default Python PATH! Suggest --download-mpi4py or --with-mpi4py-path!')
    else:
        raise RuntimeError('mpi4py unrecognized mode of building mpi4py! Suggest using --download-mpi4py!')

    if self.directory:
      installLibPath = os.path.join(self.directory, 'lib')
      if not os.path.isfile(os.path.join(installLibPath,'mpi4py','__init__.py')):
        raise RuntimeError('mpi4py not found at %s' % installLibPath)
      self.python.path.add(installLibPath)
      self.pythonpath = installLibPath

    self.addMakeMacro('MPI4PY',"yes")
    self.found = 1

