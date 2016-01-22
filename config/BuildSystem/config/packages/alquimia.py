import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    #  temporarily use a fork of Alquimia with needed changes in it. Pull request already made to alquimia developers
    self.download          = ['ssh://hg@bitbucket.org/petsc/alquimia']
    self.functions         = []
    self.includes          = []
    self.hastests          = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.petscdir       = framework.require('PETSc.options.petscdir', self.setCompilers)
    self.mpi   = framework.require('config.packages.MPI', self)
    self.hdf5  = framework.require('config.packages.hdf5', self)
    self.pflotran  = framework.require('config.packages.pflotran', self)
    self.deps  = [self.mpi, self.hdf5, self.pflotran]
    return

  def Install(self):
    return self.installDir

  def configureLibrary(self):
    ''' Just assume the downloaded library will work'''
    if self.framework.clArgDB.has_key('with-alquimia'):
      raise RuntimeError('Alquimia does not support --with-alquimia; only --download-alquimia')
    if self.framework.clArgDB.has_key('with-alquimia-dir'):
      raise RuntimeError('Alquimia does not support --with-alquimia-dir; only --download-alquimia')
    if self.framework.clArgDB.has_key('with-alquimia-include'):
      raise RuntimeError('Alquimia does not support --with-alquimia-include; only --download-alquimia')
    if self.framework.clArgDB.has_key('with-alquimia-lib'):
      raise RuntimeError('Alquimia does not support --with-alquimia-lib; only --download-alquimia')
    if self.framework.clArgDB.has_key('with-alquimia-shared'):
      raise RuntimeError('Alquimia does not support --with-alquimia-shared')

    self.checkDownload()
    self.include = [os.path.join(self.installDir,'include')]
    self.lib     = [os.path.join(self.installDir,'lib','libalquimia_c.a'),os.path.join(self.installDir,'lib','libalquimia_cutils.a'),os.path.join(self.installDir,'lib','libalquimia_fortran.a')]
    self.found   = 1
    self.dlib    = self.lib
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

  def postProcess(self):
    try:
      self.logPrintBox('Compiling Alquimia; this may take several minutes')
      generic=' COMPILER=generic '
      if config.setCompilers.Configure.isGNU(self.setCompilers.CC, self.log):
        generic=''
      if not self.compilerFlags.debugging:
        generic = generic+' RELEASE=1 '
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && '+generic+self.make.make+' PFLOTRAN_DIR='+self.installDir+' PETSC_DIR='+self.petscdir.dir+' PETSC_ARCH='+self.arch+' libs',timeout=1000, log = self.log)
      self.log.write(output+err)
      self.logPrintBox('Installing Pflotran; this may take several minutes')
      self.installDirProvider.printSudoPasswordMessage(1)
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.packageDir+' && '+self.installDirProvider.installSudo+'cp -f src/alquimia/c/*.a src/alquimia/fortran/*.a  '+os.path.join(self.installDir,'lib'),timeout=1000, log = self.log)
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.packageDir+' && '+self.installDirProvider.installSudo+'cp -f '+os.path.join('src','alquimia','fortran','*.mod')+' '+self.include[0],timeout=1000, log = self.log)
      self.log.write(output+err)
    except RuntimeError, e:
      raise RuntimeError('Error running make on Alquimia: '+str(e))


