import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download          = ['https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.3.tar.gz',
                              'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpi4py-3.0.3.tar.gz']
    self.functions         = []
    self.includes          = []
    self.useddirectly      = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.python          = framework.require('config.packages.python',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.deps            = [self.mpi]
    return

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

    logfile = os.path.join(self.petscdir.dir,self.arch,'lib','petsc','conf','mpi4py.log')
    self.framework.pushLanguage('C')
    self.logPrintBox('Building mpi4py')
    cleancmd = 'MPICC='+self.framework.getCompiler()+'  '+archflags+self.python.pyexe+' setup.py clean --all  > '+logfile+' 2>&1'
    output,err,ret  = config.base.Configure.executeShellCommand(cleancmd, cwd=self.packageDir, timeout=100, log=self.log)
    if ret: raise RuntimeError('Error cleaning mpi4py. Check '+logfile)

    buildcmd = 'MPICC='+self.framework.getCompiler()+'  '+archflags+self.python.pyexe+' setup.py build >> '+logfile+' 2>&1'
    output,err,ret  = config.base.Configure.executeShellCommand(buildcmd, cwd=self.packageDir, timeout=100, log=self.log)
    if ret: raise RuntimeError('Error building mpi4py. Check '+logfile)

    self.logPrintBox('Installing mpi4py')
    installcmd = 'MPICC='+self.framework.getCompiler()+' '+self.python.pyexe+' setup.py install --install-lib='+installLibPath+' >> '+logfile+' 2>&1'
    output,err,ret  = config.base.Configure.executeShellCommand(installcmd, cwd=self.packageDir, timeout=100, log=self.log)
    if ret: raise RuntimeError('Error installing mpi4py. Check '+logfile)
    self.framework.popLanguage()

    if 'PYTHONPATH' in os.environ:
      self.logPrintBox('To use mpi4py, do\nexport PYTHONPATH=${PYTHONPATH}'+os.pathsep+installLibPath)
      os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+os.pathsep+installLibPath
    else:
      self.logPrintBox('To use mpi4py, do\nexport PYTHONPATH='+installLibPath)
      os.environ['PYTHONPATH'] = installLibPath
    self.addMakeMacro('MPI4PY',"yes")
    self.addMakeMacro('PETSC_MPI4PY_PYTHONPATH',installLibPath)
    self.found = 1
    return self.installDir

  def configureLibrary(self):
    self.checkDownload()
    if not self.sharedLibraries.useShared:
        raise RuntimeError('mpi4py requires PETSc be built with shared libraries; rerun with --with-shared-libraries')
    if not self.python.numpy:
        raise RuntimeError('mpi4py, in the context of PETSc,requires Python with numpy module installed.\n'
                           'Please install using package managers - for ex: "apt" or "dnf" (on linux),\n'
                           'or  using: %s -m pip install %s' % (self.python.pyexe, 'numpy'))

