import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download         = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/numpy-1.0.4.tar.gz']
    self.includes         = ['']
    self.includedir       = ''
    self.libdir           = ''
    self.complex          = 0   # 0 means cannot use complex
    self.cxx              = 0   # 1 means requires C++
    self.fc               = 0   # 1 means requires fortran
    self.double           = 1   # 1 means requires double precision 
    self.requires32bitint = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.petscdir = self.framework.require('PETSc.utilities.petscdir',self)
    return

  def Install(self):
    import os, sys
    numpyDir = self.getDir()
    try:
      self.logPrintBox('Installing numpy; this may take several minutes')
      output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+numpyDir+' && python setup.py install --prefix='+self.installDir, timeout=2500, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error running setup.py on numpy: '+str(e))
    self.framework.actions.addArgument('numpy', 'Install', 'Installed numpy into '+self.installDir)
    # locate installed numpy
    installDir = os.path.abspath(self.installDir)
    for dir in os.listdir(os.path.join(installDir,'lib')):
      if dir.startswith('python') and os.path.isdir(os.path.join(installDir,'lib', dir)):
        pp = os.path.join(installDir,'lib',dir,'site-packages')
        self.logClearRemoveDirectory()
        self.logPrintBox('To use numpy add the following to your shell startup file (.cshrc, .bashrc etc)\n (csh/tcsh) setenv PYTHONPATH ${PYTHONPATH}:'+pp+'\n (sh/bash) PYTHONPATH=${PYTHONPATH}:'+pp+'; export PYTHONPATH')
        self.logResetRemoveDirectory()
    return self.installDir

  def configureLibrary(self):
    d = self.checkDownload(1)
    if d: return
    try:
      import numpy
    except:
      d = self.checkDownload(2)
      if d: return
      raise RuntimeError('Could not find numpy, either fix PYTHONPATH and rerun or use --download-numpy')
    return 
