import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download          = ['https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.3.tar.gz',
                              'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpi4py-3.0.3.tar.gz']
    self.functions         = []
    self.includes          = []
    self.useddirectly      = 0
    self.builtafterpetsc   = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.python          = framework.require('config.packages.python',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
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

    self.addMakeRule('mpi4pybuild','', \
                       ['@echo "*** Building mpi4py ***"',\
                          '@(MPICC=${PCC} && export MPICC && cd '+self.packageDir+' && \\\n\
           '+self.python.pyexe+' setup.py clean --all && \\\n\
           '+archflags+self.python.pyexe+' setup.py build ) > ${PETSC_ARCH}/lib/petsc/conf/mpi4py.log 2>&1 || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building mpi4py. Check ${PETSC_ARCH}/lib/petsc/conf/mpi4py.log" && \\\n\
             echo "********************************************************************" && \\\n\
             exit 1)'])
    self.addMakeRule('mpi4pyinstall','', \
                       ['@echo "*** Installing mpi4py ***"',\
                          '@(MPICC=${PCC} && export MPICC && cd '+self.packageDir+' && \\\n\
           '+archflags+self.python.pyexe+' setup.py install --install-lib='+installLibPath+') \\\n\
               >> ${PETSC_ARCH}/lib/petsc/conf/mpi4py.log 2>&1 || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building mpi4py. Check ${PETSC_ARCH}/lib/petsc/conf/mpi4py.log" && \\\n\
             echo "********************************************************************" && \\\n\
             exit 1)',\
                          '@echo "====================================="',\
                          '@echo "To use mpi4py, add '+installLibPath+' to PYTHONPATH"',\
                          '@echo "export PYTHONPATH=${PYTHONPATH}:"'+os.path.join(self.installDir,'lib'),\
                          '@echo "====================================="'])
    self.addMakeMacro('MPI4PY',"yes")
    if self.framework.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      self.addMakeRule('mpi4py-build','mpi4pybuild')
      self.addMakeRule('mpi4py-install','mpi4pyinstall')
    else:
      self.addMakeRule('mpi4py-build','mpi4pybuild mpi4pyinstall')
      self.addMakeRule('mpi4py-install','')

    return self.installDir

  def configureLibrary(self):
    self.checkDownload()
    if not self.sharedLibraries.useShared:
        raise RuntimeError('mpi4py requires PETSc be built with shared libraries; rerun with --with-shared-libraries')
    if not self.python.numpy:
        raise RuntimeError('mpi4py, in the context of PETSc,requires Python with numpy module installed.\n'
                           'Please install using package managers - for ex: "apt" or "dnf" (on linux),\n'
                           'or with "pip" using: %s -m pip install %s' % (self.python.pyexe, 'numpy'))

  def alternateConfigureLibrary(self):
    self.addMakeRule('mpi4py-build','')
    self.addMakeRule('mpi4py-install','')
