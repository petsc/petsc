import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/qblaslapack.gz']
    self.functions = ['ddot_']
    self.includes  = []
    self.liblist   = [['libqlapack.a','libqblas.a']]
    self.double    = 0

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    return

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
    for flag in ['-MT','-MTd','-MD','-threads']:
      if cflags.find(flag) >=0: return flag
    return ''

  def Install(self):
    import os

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
        line = 'COPTFLAGS  = '+self.setCompilers.getCompilerFlags()
        #  the f2cblaslapack source code only supports double precision
        line += ' -DDOUBLE=__float128 -DLONG=""\n'
        self.setCompilers.popLanguage()
      if line.startswith('CNOOPT'):
        self.setCompilers.pushLanguage('C')
        noopt = self.checkNoOptFlag()
        line = 'CNOOPT = '+noopt+ ' '+self.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.getPrecisionFlag(self.setCompilers.getCompilerFlags())+' '+self.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())
        #  the f2cblaslapack source code only supports double precision
        line += ' -DDOUBLE=__float128 -DLONG=""\n'
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
      self.logPrintBox('Compiling QBLASLAPACK; this may take several minutes')
      output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+blasDir+' && make -f tmpmakefile cleanblaslapck cleanlib && make -f tmpmakefile', timeout=2500, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error running make on '+blasDir+': '+str(e))
    try:
      output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+blasDir+' && mv -f libqblas.'+self.setCompilers.AR_LIB_SUFFIX+' libqlapack.'+self.setCompilers.AR_LIB_SUFFIX+' '+ libdir, timeout=30, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error moving '+blasDir+' libraries: '+str(e))

    try:
      output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+blasDir+' && cp -f tmpmakefile '+os.path.join(self.confDir, self.name), timeout=30, log = self.framework.log)
    except RuntimeError, e:
      raise RuntimeError('Error copying configure file')
    return self.installDir

  #
  # When BlasLapack.py is cleaned and the downloads in it put elsewhere then this will be tested in there and not needed here
  #
  def consistencyChecks(self):
    PETSc.package.NewPackage.consistencyChecks(self)
    self.addDefine('BLASLAPACK_UNDERSCORE',1)
    for baseName in ['gges', 'tgsen', 'gesvd','getrf','getrs','geev','gelss','syev','syevx','sygv','sygvx','getrf','potrf','getrs','potrs','stebz','pttrf','pttrs','stein','orgqr','stebz']:
      routine = 'd'+baseName+'_'
      oldLibs = self.compilers.LIBS
      if not self.libraries.check(self.lib, routine):
        self.addDefine('MISSING_LAPACK_'+baseName.upper(), 1)
      self.compilers.LIBS = oldLibs
    return
