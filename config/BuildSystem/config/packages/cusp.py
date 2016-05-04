from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download        = ['git://https://github.com/cusplibrary/cusplibrary.git']
    self.gitcommit       = '6ef9eca83df5b8774321cda07148023ae7458deb'
    self.includes        = ['cusp/version.h']
    self.includedir      = ['','include']
    self.libdir          = ''
    self.forceLanguage   = 'CUDA'
    self.cxx             = 0
    self.complex         = 0   # Currently CUSP with complex numbers is not supported
    self.CUSPVersion     = '0400' # Minimal cusp version is 0.4 
    self.CUSPVersionStr  = str(int(self.CUSPVersion)/1000) + '.' + str(int(self.CUSPVersion)%100)
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.cuda = framework.require('config.packages.cuda', self)
    self.deps   = [self.cuda]
    return

  def Install(self):
    import shutil
    import os
    self.log.write('boostDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    srcdir = os.path.join(self.packageDir,'cusp')
    destdir = os.path.join(self.installDir,'include','cusp')
    if self.installSudo:
      self.installDirProvider.printSudoPasswordMessage()
      try:
        output,err,ret  = config.base.Configure.executeShellCommand(self.installSudo+'mkdir -p '+destdir+' && '+self.installSudo+'rm -rf '+destdir+'  && '+self.installSudo+'cp -rf '+srcdir+' '+destdir, timeout=6000, log = self.log)
      except RuntimeError, e:
        raise RuntimeError('Error copying Boost files from '+os.path.join(self.packageDir, 'Boost')+' to '+packageDir)
    else:
      try:
        if os.path.isdir(destdir): shutil.rmtree(destdir)
        shutil.copytree(srcdir,destdir)
      except RuntimeError,e:
        raise RuntimeError('Error installing Boost include files: '+str(e))
    return self.installDir

  def getSearchDirectories(self):
    import os
    yield ''
    yield os.path.join('/usr','local','cuda')
    yield os.path.join('/usr','local','cuda','cusp')
    return

  def checkCUSPVersion(self):
    if 'known-cusp-version' in self.argDB:
      if self.argDB['known-cusp-version'] < self.CUSPVersion:
        raise RuntimeError('CUSP version error '+self.argDB['known-cusp-version']+' < '+self.CUSPVersion+': PETSC currently requires CUSP version '+self.CUSPVersionStr+' or higher')
    elif not self.argDB['with-batch']:
      self.pushLanguage('CUDA')
      oldFlags = self.compilers.CUDAPPFLAGS
      self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.include)
      if not self.checkRun('#include <cusp/version.h>\n#include <stdio.h>', 'if (CUSP_VERSION < ' + self.CUSPVersion +') {printf("Invalid version %d\\n", CUSP_VERSION); return 1;}'):
        raise RuntimeError('CUSP version error: PETSC currently requires CUSP version '+self.CUSPVersionStr+' or higher.')
      self.compilers.CUDAPPFLAGS = oldFlags
      self.popLanguage()
    else:
      raise RuntimeError('Batch configure does not work with CUDA\nOverride all CUDA configuration with options, such as --known-cusp-version')
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does a additional tests needed by CUSP'''
    config.package.Package.configureLibrary(self)
    self.checkCUSPVersion()
    return

