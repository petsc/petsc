import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
#    self.hghash            = '611092f80ddb'
    self.download          = ['hg://https://bitbucket.org/pflotran/pflotran-xsdk']
    self.functions         = []
    self.includes          = []
    self.hastests          = 1
    self.fc                = 1    # 1 means requires fortran
    self.linkedbypetsc     = 0
    self.useddirectly      = 0
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi      = framework.require('config.packages.MPI', self)
    self.hdf5     = framework.require('config.packages.hdf5', self)
    self.deps     = [self.mpi, self.hdf5]
    return

  def Install(self):
    return self.installDir

  def configureLibrary(self):
    ''' Just assume the downloaded library will work'''
    if self.framework.clArgDB.has_key('with-pflotran'):
      raise RuntimeError('Pflotran does not support --with-pflotran; only --download-pflotran')
    if self.framework.clArgDB.has_key('with-pflotran-dir'):
      raise RuntimeError('Pflotran does not support --with-pflotran-dir; only --download-pflotran')
    if self.framework.clArgDB.has_key('with-pflotran-include'):
      raise RuntimeError('Pflotran does not support --with-pflotran-include; only --download-pflotran')
    if self.framework.clArgDB.has_key('with-pflotran-lib'):
      raise RuntimeError('Pflotran does not support --with-pflotran-lib; only --download-pflotran')
    if self.framework.clArgDB.has_key('with-pflotran-shared'):
      raise RuntimeError('Pflotran does not support --with-pflotran-shared')

    self.checkDownload()
    self.include = [os.path.join(self.installDir,'include')]
    self.lib     = [os.path.join(self.installDir,'lib','libpflotranchem.a')]
    self.found   = 1
    self.dlib    = self.lib
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

  def postProcess(self):
    self.compilePETSc()

    try:
      self.logPrintBox('Configure Pflotran; this may take several minutes')
      if self.framework.argDB['prefix']:
        PDIR   = 'PETSC_DIR='+self.framework.argDB['prefix']
        PARCH  = ''
        PREFIX = '--prefix='+self.framework.argDB['prefix']
      else:
        PDIR   = 'PETSC_DIR='+self.petscdir.dir
        PARCH  = 'PETSC_ARCH='+self.arch
        PREFIX = '--prefix='+os.path.join(self.petscdir.dir,self.arch)
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.packageDir+' && '+PARCH+' '+PDIR+' ./configure all '+PREFIX,timeout=10, log = self.log)
      self.log.write(output+err)

      self.logPrintBox('Compiling Pflotran; this may take several minutes')
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.packageDir+' && make all',timeout=1000, log = self.log)
      self.log.write(output+err)

      self.logPrintBox('Installing Pflotran; this may take several minutes')
      self.installDirProvider.printSudoPasswordMessage(1)
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.packageDir+' && '+self.installDirProvider.installSudo+' make install',timeout=100, log = self.log)
      self.log.write(output+err)
    except RuntimeError, e:
      raise RuntimeError('Error configuring/compiling or installing Pflotran: '+str(e))


