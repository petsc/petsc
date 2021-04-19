import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit         = 'v2.2-p4'
    self.download          = ['git://https://bitbucket.org/petsc/pkg-chaco.git',
                              'https://bitbucket.org/petsc/pkg-chaco/get/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames  = ['petsc-pkg-chaco','Chaco']
    self.functions         = ['interface']
    self.includes          = [] #Chaco does not have an include file
    self.liblist           = [['libchaco.a']]
    self.license           = 'http://www.cs.sandia.gov/web1400/1400_download.html'
    self.downloadonWindows = 1
    self.requires32bitint  = 1;  # 1 means that the package will not work with 64 bit integers
    self.hastests          = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.mathlib        = framework.require('config.packages.mathlib',self)
    self.deps           = [self.mathlib]
    return

  def Install(self):
    import os, glob
    self.log.write('chacoDir = '+self.packageDir+' installDir '+self.installDir+'\n')

    mkfile = 'make.inc'
    g = open(os.path.join(self.packageDir, mkfile), 'w')
    self.pushLanguage('C')
    g.write('CC = '+self.getCompiler()+'\n')
    g.write('CFLAGS = '+self.updatePackageCFlags(self.getCompilerFlags())+'\n')
    g.write('OFLAGS = '+self.updatePackageCFlags(self.getCompilerFlags())+'\n')
    self.popLanguage()
    g.close()

    if self.installNeeded(mkfile):
      try:
        self.logPrintBox('Compiling and installing chaco; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret  = config.package.Package.executeShellCommandSeq(
          ['make clean',
           'make',
           self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+' '+'libchaco.'+
           self.setCompilers.AR_LIB_SUFFIX+' `ls */*.o |grep -v main/main.o`',
           self.setCompilers.RANLIB+' libchaco.'+self.setCompilers.AR_LIB_SUFFIX,
           self.installSudo+'mkdir -p '+os.path.join(self.installDir,self.libdir),
           self.installSudo+'cp libchaco.'+self.setCompilers.AR_LIB_SUFFIX+' '+os.path.join(self.installDir,self.libdir)
          ], cwd=os.path.join(self.packageDir, 'code'), timeout=2500, log = self.log)

      except RuntimeError as e:
        raise RuntimeError('Error running make on CHACO: '+str(e))
      self.postInstall(output+err, mkfile)
    return self.installDir

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    if not self.libraries.check(self.lib, 'ddot_chaco',otherLibs=self.mathlib.lib):
      raise RuntimeError('You cannot use Chaco package from Sandia as it contains an incorrect ddot() routine that conflicts with BLAS\nUse --download-chaco')

