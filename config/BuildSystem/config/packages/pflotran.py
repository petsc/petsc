import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = 'xsdk-0.2.0-rc1'
    self.download          = ['git://https://bitbucket.org/pflotran/pflotran']
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
    self.deps     = [self.mpi]
    self.odeps    = [self.hdf5]
    return

  # the install is delayed until postProcess() since pflotran install requires PETSc to be installed before pflotran can be built
  def Install(self):
    return self.installDir

  def configureLibrary(self):
    ''' Since pflotran cannot be built until after PETSc is compiled we need to just assume the downloaded library will work'''
    if 'with-pflotran' in self.framework.clArgDB:
      raise RuntimeError('Pflotran does not support --with-pflotran; only --download-pflotran')
    if 'with-pflotran-dir' in self.framework.clArgDB:
      raise RuntimeError('Pflotran does not support --with-pflotran-dir; only --download-pflotran')
    if 'with-pflotran-include' in self.framework.clArgDB:
      raise RuntimeError('Pflotran does not support --with-pflotran-include; only --download-pflotran')
    if 'with-pflotran-lib' in self.framework.clArgDB:
      raise RuntimeError('Pflotran does not support --with-pflotran-lib; only --download-pflotran')
    if 'with-pflotran-shared' in self.framework.clArgDB:
      raise RuntimeError('Pflotran does not support --with-pflotran-shared')

    self.checkDownload()
    self.include = [os.path.join(self.installDir,'include')]
    self.lib     = [os.path.join(self.installDir,'lib','libpflotranchem.a'),os.path.join(self.installDir,'lib','libpflotran.a')]
    self.found   = 1
    self.dlib    = self.lib
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

  def postProcess(self):
    self.compilePETSc()

    try:
      self.logPrintBox('Configure Pflotran; this may take several minutes')
      # TODO: remove this prefix code and use the mechanisms in package.py for selecting the destination directory; currently if postProcess is used
      # TODO: package.py may not allow installing in prefix location
      if self.framework.argDB['prefix']:
        PDIR   = 'PETSC_DIR='+self.framework.argDB['prefix']
        PARCH  = ''
        PREFIX = '--prefix='+self.framework.argDB['prefix']
      else:
        PDIR   = 'PETSC_DIR='+self.petscdir.dir
        PARCH  = 'PETSC_ARCH='+self.arch
        PREFIX = '--prefix='+os.path.join(self.petscdir.dir,self.arch)
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.packageDir+' && '+PARCH+' '+PDIR+' ./configure all '+PREFIX,timeout=60, log = self.log)
      self.log.write(output+err)

      self.logPrintBox('Compiling Pflotran; this may take several minutes')
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.packageDir+' && make all',timeout=1000, log = self.log)
      self.log.write(output+err)

      self.logPrintBox('Installing Pflotran; this may take several minutes')
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.packageDir+' && make install',timeout=100, log = self.log)
      self.log.write(output+err)
    except RuntimeError as e:
      raise RuntimeError('Error configuring/compiling or installing Pflotran: '+str(e))


