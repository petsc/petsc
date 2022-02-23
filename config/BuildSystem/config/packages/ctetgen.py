import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = 'ctetgen-0.10'
    self.download          = ['git://https://bitbucket.org/petsc/ctetgen','https://bitbucket.org/petsc/ctetgen/get/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames  = ['ctetgen','petsc-ctetgen']
    self.functions         = []
    self.includes          = []
    self.hastests          = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    return

  # the install of ctetgen is delayed until postProcess() since ctetgen install requires PETSc to have created its build/makefiles before installing
  # ctetgen can (and is) built and installed before the PETSc source code and libraries are made
  def Install(self):
    return self.installDir

  def configureLibrary(self):
    '''Since ctetgen cannot be built until after PETSc configure is complete we need to just assume the downloaded library will work'''
    if 'with-ctetgen' in self.framework.clArgDB:
      raise RuntimeError('Ctetgen does not support --with-ctetgen; only --download-ctetgen')
    if 'with-ctetgen-dir' in self.framework.clArgDB:
      self.ctetgenDir = self.framework.argDB['with-ctetgen-dir']
    if 'with-ctetgen-include' in self.framework.clArgDB:
      raise RuntimeError('Ctetgen does not support --with-ctetgen-include; only --download-ctetgen')
    if 'with-ctetgen-lib' in self.framework.clArgDB:
      raise RuntimeError('Ctetgen does not support --with-ctetgen-lib; only --download-ctetgen')
    if 'with-ctetgen-shared' in self.framework.clArgDB:
      raise RuntimeError('Ctetgen does not support --with-ctetgen-shared')

    if not hasattr(self,'ctetgenDir'):
      self.checkDownload()
      self.ctetgenDir = self.installDir
    self.include = [os.path.join(self.ctetgenDir,'include')]
    self.lib     = [os.path.join(self.ctetgenDir,'lib','libctetgen.a')]
    self.found   = 1
    self.dlib    = self.lib
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

  def postProcess(self):
    if not hasattr(self,'installDir'):
      return
    try:
      self.logPrintBox('Compiling Ctetgen; this may take several minutes')
      # uses the regular PETSc library builder and then moves result
      # turn off any compiler optimizations as they may break CTETGEN
      self.pushLanguage('C')
      cflags = self.checkNoOptFlag()+' '+self.getSharedFlag(self.getCompilerFlags())+' '+self.getPointerSizeFlag(self.getCompilerFlags())+' '+self.getWindowsNonOptFlags(self.getCompilerFlags())+' '+self.getDebugFlags(self.getCompilerFlags())
      self.popLanguage()
      output,err,ret  = config.package.GNUPackage.executeShellCommand(self.make.make+' PETSC_DIR='+self.petscdir.dir+' clean lib PCC_FLAGS="'+cflags+'"',timeout=1000, log = self.log, cwd=self.packageDir)
      self.log.write(output+err)
      self.logPrintBox('Installing Ctetgen; this may take several minutes')
      output,err,ret  = config.package.GNUPackage.executeShellCommand(self.make.make+' PETSC_DIR='+self.petscdir.dir+' prefix='+self.installDir+' install-ctetgen',timeout=1000, log = self.log, cwd=self.packageDir)
      self.log.write(output+err)
    except RuntimeError as e:
      raise RuntimeError('Error running make on Ctetgen: '+str(e))
