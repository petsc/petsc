import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download          = ['http://mpi4py.googlecode.com/files/mpi4py-1.2.2.tar.gz']
    self.functions         = []
    self.includes          = []
    self.liblist           = []
    self.complex           = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.numpy           = framework.require('PETSc.packages.Numpy',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.utilities.sharedLibraries', self)    
    self.petscconfigure  = framework.require('PETSc.Configure',self)
    return

  def Install(self):
    import os
    pp = os.path.join(self.installDir,'lib','python*','site-packages')
    if self.setCompilers.isDarwin():
      apple = 'You may need to\n (csh/tcsh) setenv MACOSX_DEPLOYMENT_TARGET 10.X\n (sh/bash) MACOSX_DEPLOYMENT_TARGET=10.X; export MACOSX_DEPLOYMENT_TARGET\nbefore running make on PETSc'
    else:
      apple = ''
    self.logClearRemoveDirectory()
    self.logResetRemoveDirectory()
    archflags = ""
    if self.setCompilers.isDarwin():
      if self.types.sizes['known-sizeof-void-p'] == 32:
        archflags = "ARCHFLAGS=\'-arch i386\'"
      else:
        archflags = "ARCHFLAGS=\'-arch x86_64\'"
    self.addMakeRule('mpi4py','', \
                       ['@MPICC=${PCC}; export MPICC; cd '+self.packageDir+';python setup.py clean --all; '+archflags+' python setup.py install --install-lib='+os.path.join(self.installDir,'lib'),\
                          '@echo "====================================="',\
                          '@echo "To use mpi4py, add '+os.path.join(self.petscconfigure.installdir,'lib')+' to PYTHONPATH"',\
                          '@echo "====================================="'])
    
    return self.installDir

  def configureLibrary(self):
    self.checkDownload(1)
    if not self.sharedLibraries.useShared:
        raise RuntimeError('mpi4py requires PETSc be built with shared libraries; rerun with --with-shared-libraries')

  def alternateConfigureLibrary(self):
    self.addMakeRule('mpi4py','')   
    self.addMakeRule('mpi4py_noinstall','')
