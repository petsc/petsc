import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit         = '3.7.0'
    self.download          = ['git://https://bitbucket.org/petsc/petsc4py','https://bitbucket.org/petsc/petsc4py/downloads/petsc4py-'+self.gitcommit+'.tar.gz']
    self.functions         = []
    self.includes          = []
    self.skippackagewithoptions = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.numpy           = framework.require('config.packages.Numpy',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
    return

  def Install(self):
    import os
    pp = os.path.join(self.installDir,'lib','python*','site-packages')
    if self.setCompilers.isDarwin(self.log):
      apple = 'You may need to\n (csh/tcsh) setenv MACOSX_DEPLOYMENT_TARGET 10.X\n (sh/bash) MACOSX_DEPLOYMENT_TARGET=10.X; export MACOSX_DEPLOYMENT_TARGET\nbefore running make on PETSc'
    else:
      apple = ''
    self.logClearRemoveDirectory()
    self.logResetRemoveDirectory()
    archflags = ""
    if self.setCompilers.isDarwin(self.log):
      if self.types.sizes['known-sizeof-void-p'] == 4:
        archflags = "ARCHFLAGS=\'-arch i386\' "
      else:
        archflags = "ARCHFLAGS=\'-arch x86_64\' "

    # if installing prefix location then need to set new value for PETSC_DIR/PETSC_ARCH
    if self.argDB['prefix']:
       newdir = 'PETSC_DIR='+os.path.abspath(os.path.expanduser(self.argDB['prefix']))+' '+'PETSC_ARCH= MPICC=${PCC} '
    else:
       newdir = 'MPICC=${PCC} '

    #  if installing as Superuser than want to return to regular user for clean and build
    if self.installSudo:
       newuser = self.installSudo+' -u $${SUDO_USER} '
    else:
       newuser = ''

    self.addMakeRule('petsc4pybuild','', \
                       ['@echo "*** Building petsc4py ***"',\
                          '@${RM} -f ${PETSC_ARCH}/lib/petsc/conf/petsc4py.errorflg',\
                          '@(cd '+self.packageDir+' && \\\n\
           '+newuser+newdir+'python setup.py clean --all && \\\n\
           '+newuser+newdir+archflags+'python setup.py build ) > ${PETSC_ARCH}/lib/petsc/conf/petsc4py.log 2>&1 || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building petsc4py. Check ${PETSC_ARCH}/lib/petsc/conf/petsc4py.log" && \\\n\
             echo "********************************************************************" && \\\n\
             touch ${PETSC_ARCH}/lib/petsc/conf/petsc4py.errorflg && \\\n\
             exit 1)'])
    self.addMakeRule('petsc4pyinstall','', \
                       ['@echo "*** Installing petsc4py ***"',\
                          '@(MPICC=${PCC} && export MPICC && cd '+self.packageDir+' && \\\n\
           '+newdir+archflags+'python setup.py install --install-lib='+os.path.join(self.installDir,'lib')+') >> ${PETSC_ARCH}/lib/petsc/conf/petsc4py.log 2>&1 || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building petsc4py. Check ${PETSC_ARCH}/lib/petsc/conf/petsc4py.log" && \\\n\
             echo "********************************************************************" && \\\n\
             exit 1)',\
                          '@echo "====================================="',\
                          '@echo "To use petsc4py, add '+os.path.join(self.installdir.dir,'lib')+' to PYTHONPATH"',\
                          '@echo "====================================="'])
    if self.argDB['prefix']:
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
    self.checkDownload()
    if self.setCompilers.isDarwin(self.log):
      # The name of the Python library on Apple is Python which does not end in the expected .dylib
      # Thus see if the python library in the standard locations points to the Python version
      import sys
      import os
      prefix = sys.exec_prefix
      if os.path.isfile(os.path.join(prefix,'Python')):
        for i in ['/usr/lib/libpython.dylib','/usr/lib/libpython'+sys.version[:3]+'.dylib','/opt/local/lib/libpython.dylib','/opt/local/lib/libpython'+sys.version[:3]+'.dylib']:
          if os.path.realpath(i) == os.path.join(prefix,'Python'):
            self.addDefine('PYTHON_LIB','"'+os.path.join(i)+'"')
            return
      if os.path.isfile(os.path.join(prefix,'lib','libpython.dylib')):
        self.addDefine('PYTHON_LIB','"'+os.path.join(prefix,'lib','libpython.dylib')+'"')
        return
      if os.path.isfile(os.path.join(prefix,'lib','libpython'+sys.version[:3]+'.dylib')):
        self.addDefine('PYTHON_LIB','"'+os.path.join(prefix,'lib','libpython'+sys.version[:3]+'.dylib')+'"')
        return
      raise RuntimeError('Unable to find Python dynamic library at prefix '+prefix)

  def alternateConfigureLibrary(self):
    self.addMakeRule('petsc4py-build','')
    self.addMakeRule('petsc4py-install','')

  def gitPreReqCheck(self):
    ''' petsc4py git download requires Cython'''
    flg = True
    try:
      import Cython
    except:
      flg = False
    return flg
