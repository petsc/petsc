import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download         = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/f2cblaslapack-3.1.1.q.tar.gz']
    self.double           = 0
    self.worksonWindows   = 1
    self.downloadonWindows= 1

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    return

  def getDefaultPrecision(self):
    '''The precision of the library'''
    if hasattr(self, 'precisionProvider'):
      if hasattr(self.precisionProvider, 'precision'):
        return self.precisionProvider.precision
    return self._defaultPrecision
  def setDefaultPrecision(self, defaultPrecision):
    '''The precision of the library'''
    self._defaultPrecision = defaultPrecision
    return
  defaultPrecision = property(getDefaultPrecision, setDefaultPrecision, doc = 'The precision of the library')

  def checkNoOptFlag(self):
    flag = '-O0'
    if self.setCompilers.checkCompilerFlag(flag): return flag
    return ''

  def getSharedFlag(self,cflags):
    for flag in ['-PIC', '-fPIC', '-KPIC', '-qpic']:
      if cflags.find(flag) >=0: return flag
    return ''

  def getPrecisionFlag(self,cflags):
    for flag in ['-m32', '-m64', '-xarch=v9','-q64']:
      if cflags.find(flag) >=0: return flag
    return ''

  def getWindowsNonOptFlags(self,cflags):
    for flag in ['-MT','-MTd','-MD','-MDd','-threads']:
      if cflags.find(flag) >=0: return flag
    return ''

  def Install(self):
    import os

    make_target = 'single double'
    if self.defaultPrecision == '__float128': make_target += ' quad'

    libdir = self.libDir
    confdir = self.confDir
    blasDir = self.packageDir

    g = open(os.path.join(blasDir,'tmpmakefile'),'w')
    f = open(os.path.join(blasDir,'makefile'),'r')
    line = f.readline()
    while line:
      if line.startswith('CC  '):
        cc = self.compilers.CC
        line = 'CC = '+cc+'\n'
      if line.startswith('COPTFLAGS '):
        self.setCompilers.pushLanguage('C')
        line = 'COPTFLAGS  = '+self.setCompilers.getCompilerFlags()+'\n'
        self.setCompilers.popLanguage()
      if line.startswith('CNOOPT'):
        self.setCompilers.pushLanguage('C')
        noopt = self.checkNoOptFlag()
        line = 'CNOOPT = '+noopt+ ' '+self.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.getPrecisionFlag(self.setCompilers.getCompilerFlags())+' '+self.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())+'\n'
        self.setCompilers.popLanguage()
      if line.startswith('AR  '):
        line = 'AR      = '+self.setCompilers.AR+'\n'
      if line.startswith('AR_FLAGS  '):
        line = 'AR_FLAGS      = '+self.setCompilers.AR_FLAGS+'\n'
      if line.startswith('LIB_SUFFIX '):
        line = 'LIB_SUFFIX = '+self.setCompilers.AR_LIB_SUFFIX+'\n'
      if line.startswith('RANLIB  '):
        line = 'RANLIB = '+self.setCompilers.RANLIB+'\n'
      if line.startswith('RM  '):
        line = 'RM = '+self.programs.RM+'\n'

      if line.startswith('include'):
        line = '\n'
      g.write(line)
      line = f.readline()
    f.close()
    g.close()

    if not self.installNeeded('tmpmakefile'): return self.installDir

    try:
      self.logPrintBox('Compiling BLASLAPACK; this may take several minutes')
      output,err,ret  = config.base.Configure.executeShellCommand('cd '+blasDir+' && make -f tmpmakefile cleanblaslapck cleanlib && make -f tmpmakefile '+make_target, timeout=2500, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error running make on '+blasDir+': '+str(e))
    try:
      output,err,ret  = config.base.Configure.executeShellCommand('cd '+blasDir+' && mv -f libf2cblas.'+self.setCompilers.AR_LIB_SUFFIX+' libf2clapack.'+self.setCompilers.AR_LIB_SUFFIX+' '+ libdir, timeout=30, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error moving '+blasDir+' libraries: '+str(e))

    try:
      output,err,ret  = config.base.Configure.executeShellCommand('cd '+blasDir+' && cp -f tmpmakefile '+os.path.join(self.confDir, self.name), timeout=30, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error copying configure file')
    return self.installDir

