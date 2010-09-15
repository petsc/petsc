import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download          = ['http://petsc4py.googlecode.com/files/petsc4py-1.1.1.tar.gz']
    self.functions         = []
    self.includes          = []
    self.liblist           = []
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.petscconfigure  = framework.require('PETSc.Configure',self)
    self.numpy           = framework.require('PETSc.packages.Numpy',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.utilities.sharedLibraries', self)
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
    if self.framework.argDB['prefix']:
      arch = ''
      self.addMakeRule('petsc4py_noinstall','')
    else:
      arch = self.arch
      self.addMakeRule('petsc4py_noinstall','petsc4py')
    archflags = ""
    if self.setCompilers.isDarwin():
      if self.types.sizes['known-sizeof-void-p'] == 32:
        archflags = "ARCHFLAGS=\'-arch i386\'"
      else:
        archflags = "ARCHFLAGS=\'-arch x86_64\'"
    self.addMakeRule('petsc4py','', \
                       ['@cd '+self.packageDir+';python setup.py clean --all; '+archflags+' python setup.py install --install-lib='+os.path.join(self.installDir,'lib'),\
                          '@echo "====================================="',\
                          '@echo "To use petsc4py, add '+os.path.join(self.petscconfigure.installdir,'lib')+' to PYTHONPATH"',\
                          '@echo "====================================="'])      
    
    return self.installDir

  def configureLibrary(self):
    if not self.sharedLibraries.useShared:
        raise RuntimeError('petsc4py requires PETSc be built with shared libraries; rerun with --with-shared')
    self.checkDownload(1)
    if self.setCompilers.isDarwin():
      # The name of the Python library on Apple is Python which does not end in the expected .dylib
      # Thus see if the python library in the standard locations points to the Python version
      import sys
      import os
      prefix = sys.exec_prefix
      if os.path.isfile(os.path.join(prefix,'Python')):
        for i in ['/usr/lib/libpython.dylib','/opt/local/lib/libpython2.5.dylib','/opt/local/lib/libpython2.6.dylib']:
          if os.path.realpath(i) == os.path.join(prefix,'Python'):
            self.addDefine('PYTHON_LIB','"'+os.path.join(i)+'"')
            return
        raise RuntimeError('realpath of /usr/lib/libpython.dylib ('+os.path.realpath('/usr/lib/libpython.dylib')+') does not point to Python library path ('+os.path.join(prefix,'Python')+') for current Python;\n Are you not using the Apple python?')
      elif os.path.isfile(os.path.join(prefix,'lib','libpython.dylib')):
        self.addDefine('PYTHON_LIB','"'+os.path.join(prefix,'lib','libpython.dylib')+'"')
      else:
        raise RuntimeError('Unable to find Python dynamic library at prefix '+prefix)

  def alternateConfigureLibrary(self):
    self.addMakeRule('petsc4py','')   
    self.addMakeRule('petsc4py_noinstall','')
      
