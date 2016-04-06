import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.hghash            = '611092f80ddb'
    self.download          = ['hg://https://bitbucket.org/pflotran/pflotran-dev']
    self.functions         = []
    self.includes          = []
    self.hastests          = 1
    self.fc                = 1    # 1 means requires fortran   
    self.linkedbypetsc     = 0
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

    # Patch the PETSc paths so that older versions of PFlotran can find Fortran include files and configuration files
    try:
      if not os.path.isdir(os.path.join(self.petscdir.dir,'include','finclude')):
        output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+os.path.join(self.petscdir.dir,'include')+' && ln -s petsc/finclude finclude',timeout=10, log = self.log)
      if not os.path.isdir(os.path.join(self.petscdir.dir,'conf')):
        output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.petscdir.dir+' && ln -s lib/petsc/conf conf',timeout=10, log = self.log)
    except RuntimeError, e:
      raise RuntimeError('Unable to make links required by older versions of PFlotran')
    try:
      self.logPrintBox('Compiling Pflotran; this may take several minutes')
      # uses the regular PETSc library builder and then moves result 
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'src','pflotran')+' && '+self.make.make+' have_hdf5=1 use_matseqaij_fix=1 PETSC_DIR='+self.petscdir.dir+' PETSC_ARCH='+self.arch+' libpflotranchem.a',timeout=1000, log = self.log)
      self.log.write(output+err)
      self.logPrintBox('Installing Pflotran; this may take several minutes')
      self.installDirProvider.printSudoPasswordMessage(1)
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.packageDir+' && '+self.installDirProvider.installSudo+'cp -f '+os.path.join('src','pflotran','libpflotran*.a')+' '+os.path.join(self.installDir,'lib'),timeout=1000, log = self.log)
      self.log.write(output+err)
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.packageDir+' && '+self.installDirProvider.installSudo+'cp -f '+os.path.join('src','pflotran','*.mod')+' '+self.include[0],timeout=100, log = self.log)
      self.log.write(output+err)
    except RuntimeError, e:
      raise RuntimeError('Error running make on Pflotran: '+str(e))


