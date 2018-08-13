import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = '8ca66eb'
    self.download          = ['git://https://bitbucket.org/petsc/ctetgen.git','http://ftp.mcs.anl.gov/pub/petsc/externalpackages/ctetgen-0.4.tar.gz']
    self.functions         = []
    self.includes          = []
    self.hastests          = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    return

  # the install is delayed until postProcess() since ctetgen install requires PETSc to have created its build/makefiles before installing
  # note that ctetgen can (and is) built before PETSc is built.
  def Install(self):
    return self.installDir

  def configureLibrary(self):
    ''' Since ctergen cannot be built until after PETSc configure is complete we need to just assume the downloaded library will work'''
    if 'with-ctetgen' in self.framework.clArgDB:
      raise RuntimeError('Ctetgen does not support --with-ctetgen; only --download-ctetgen')
    if 'with-ctetgen-dir' in self.framework.clArgDB:
      raise RuntimeError('Ctetgen does not support --with-ctetgen-dir; only --download-ctetgen')
    if 'with-ctetgen-include' in self.framework.clArgDB:
      raise RuntimeError('Ctetgen does not support --with-ctetgen-include; only --download-ctetgen')
    if 'with-ctetgen-lib' in self.framework.clArgDB:
      raise RuntimeError('Ctetgen does not support --with-ctetgen-lib; only --download-ctetgen')
    if 'with-ctetgen-shared' in self.framework.clArgDB:
      raise RuntimeError('Ctetgen does not support --with-ctetgen-shared')

    self.checkDownload()
    self.include = [os.path.join(self.installDir,'include')]
    self.lib     = [os.path.join(self.installDir,'lib','libctetgen.a')]
    self.found   = 1
    self.dlib    = self.lib
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

  def postProcess(self):
    try:
      self.logPrintBox('Compiling Ctetgen; this may take several minutes')
      # uses the regular PETSc library builder and then moves result 
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.packageDir+' && '+self.make.make+' PETSC_DIR='+self.petscdir.dir+' clean lib',timeout=1000, log = self.log)
      self.log.write(output+err)
      self.logPrintBox('Installing Ctetgen; this may take several minutes')
      self.installDirProvider.printSudoPasswordMessage(1)
      output,err,ret  = config.package.GNUPackage.executeShellCommand('cd '+self.packageDir+' && '+self.installDirProvider.installSudo+self.make.make+' PETSC_DIR='+self.petscdir.dir+' install-ctetgen',timeout=1000, log = self.log)
      self.log.write(output+err)
    except RuntimeError as e:
      raise RuntimeError('Error running make on Ctetgen: '+str(e))


