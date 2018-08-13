from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download        = ['git://https://github.com/cusplibrary/cusplibrary.git']
    self.gitcommit       = 'v0.5.1'
    self.includes        = ['cusp/version.h']
    self.includedir      = ['','include']
    self.libdir          = ''
    self.forceLanguage   = 'CUDA'
    self.cxx             = 0
    self.complex         = 0   # Currently CUSP with complex numbers is not supported
    self.CUSPVersion     = ''
    self.CUSPMinVersion  = '500' # Minimal cusp version is 0.5.0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.cuda = framework.require('config.packages.cuda', self)
    self.deps   = [self.cuda]
    return

  def Install(self):
    import shutil
    import os
    self.log.write('cuspDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    srcdir = os.path.join(self.packageDir,'cusp')
    destdir = os.path.join(self.installDir,'include','cusp')
    if self.installSudo:
      self.installDirProvider.printSudoPasswordMessage()
      try:
        output,err,ret  = config.base.Configure.executeShellCommand(self.installSudo+'mkdir -p '+destdir+' && '+self.installSudo+'rm -rf '+destdir+'  && '+self.installSudo+'cp -rf '+srcdir+' '+destdir, timeout=6000, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error copying Cusp files from '+os.path.join(self.packageDir, 'Cusp')+' to '+packageDir)
    else:
      try:
        if os.path.isdir(destdir): shutil.rmtree(destdir)
        shutil.copytree(srcdir,destdir)
      except RuntimeError as e:
        raise RuntimeError('Error installing Cusp include files: '+str(e))
    return self.installDir

  def getSearchDirectories(self):
    import os
    yield ''
    yield os.path.join('/usr','local','cuda')
    yield os.path.join('/usr','local','cuda','cusp')
    return

  def verToStr(self,ver):
    return str(int(ver)/100000) + '.' + str(int(ver)/100%1000) + '.' + str(int(ver)% 100)

  def checkCUSPVersion(self):
    import re
    HASHLINESPACE = ' *(?:\n#.*\n *)*'
    self.pushLanguage('CUDA')
    oldFlags = self.compilers.CUDAPPFLAGS
    self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.include)
    cusp_test = '#include <cusp/version.h>\nint cusp_ver = CUSP_VERSION;\n'
    if self.checkCompile(cusp_test):
      buf = self.outputPreprocess(cusp_test)
      try:
        self.CUSPVersion = re.compile('\nint cusp_ver ='+HASHLINESPACE+'([0-9]+)'+HASHLINESPACE+';').search(buf).group(1)
      except:
        self.logPrint('Unable to parse CUSP version from header. Probably a buggy preprocessor')
    self.compilers.CUDAPPFLAGS = oldFlags
    self.popLanguage()
    if self.CUSPVersion and self.CUSPVersion < self.CUSPMinVersion:
      raise RuntimeError('CUSP version error: PETSC currently requires CUSP version '+self.verToStr(self.CUSPMinVersion)+' or higher. Found version '+self.verToStr(self.CUSPVersion))
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does a additional tests needed by CUSP'''
    config.package.Package.configureLibrary(self)
    self.checkCUSPVersion()
    return

  def configure(self):
    if self.cuda.CUDAVersion >= '9000':
      self.gitcommit = '116b090' # cusp/cuda9 branch
    return config.package.Package.configure(self)

