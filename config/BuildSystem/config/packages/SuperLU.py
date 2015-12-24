import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit        = 'v5.1'
    self.download         = ['git://https://bitbucket.org/petsc/pkg-superlu.git',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/superlu_5.1.tar.gz']
    self.functions        = ['set_default_options']
    self.includes         = ['slu_ddefs.h']
    self.liblist          = [['libsuperlu_5.1.a']]
    # SuperLU has NO support for 64 bit integers, use SuperLU_Dist if you need that
    self.requires32bitint = 1;  # 1 means that the package will not work with 64 bit integers
    self.excludedDirs     = ['SuperLU_DIST','SuperLU_MT']
    # SuperLU does not work with --download-fblaslapack with Compaqf90 compiler on windows.
    # However it should work with intel ifort.
    self.downloadonWindows= 1
    self.hastests         = 1
    self.hastestsdatafiles= 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = self.framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return

  def Install(self):
    import os
    if (self.compilers.c99flag == None):
      raise RuntimeError(self.PACKAGE+': install requires c99 compiler. Configure cold not determine compatilbe compiler flag. Perhaps you can specify via CFLAG')
    if not self.make.haveGNUMake:
      raise RuntimeError(self.PACKAGE+': install requires GNU make. Suggest using --download-make')

    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    g.write('SuperLUroot  = '+self.packageDir+'\n')
    g.write('TMGLIB       = tmglib.'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('SUPERLULIB   = $(SuperLUroot)/lib/libsuperlu_5.1.'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('BLASLIB      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('BLASDEF      = -DUSE_VENDOR_BLAS\n')
    g.write('LIBS         = $(SUPERLULIB) $(BLASLIB)\n')
    g.write('ARCH         = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS       = '+self.setCompilers.getCompilerFlags()+'\n')
    g.write('LOADER       = '+self.setCompilers.getLinker()+'\n')
    g.write('LOADOPTS     = \n')
    self.setCompilers.popLanguage()

    # set blas name mangling
    if self.blasLapack.mangling == 'underscore':
      g.write('CDEFS        = -DAdd_')
    elif self.blasLapack.mangling == 'caps':
      g.write('CDEFS   = -DUpCase')
    else:
      g.write('CDEFS   = -DNoChange')
    g.write('\n')

    g.write('MATLAB       =\n')
    g.write('NOOPTS       = '+self.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.getPointerSizeFlag(self.setCompilers.getCompilerFlags())+' '+self.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())+'\n')
    g.close()
    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling and installing superlu; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'lib'), timeout=2500, log=self.log)
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'include'), timeout=2500, log=self.log)
        if not os.path.exists(os.path.join(self.packageDir,'lib')):
          os.makedirs(os.path.join(self.packageDir,'lib'))
        output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+' && '+self.make.make+' clean && '+self.make.make+' superlulib && '+self.installSudo+'cp -f lib/*.'+self.setCompilers.AR_LIB_SUFFIX+' '+os.path.join(self.installDir,self.libdir,'')+' &&  '+self.installSudo+'cp -f SRC/*.h '+os.path.join(self.installDir,self.includedir,''), timeout=2500, log = self.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUPERLU: '+str(e))
      self.postInstall(output+err,'make.inc')
    return self.installDir
