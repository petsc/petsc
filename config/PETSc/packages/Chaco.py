import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download     = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/Chaco-2.2.tar.gz']
    self.functions    = ['interface']
    self.includes     = [] #Chaco does not have an include file
    self.needsMath    = 1
    self.complex      = 1
    self.liblist      = [['libchaco.a']]
    self.license      = 'http://www.cs.sandia.gov/~web9200/9200_download.html'
    return

  def Install(self):
    import os
    self.framework.log.write('chacoDir = '+self.packageDir+' installDir '+self.installDir+'\n')

    mkfile = 'make.inc'
    g = open(os.path.join(self.packageDir, mkfile), 'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS = '+self.setCompilers.getCompilerFlags()+'\n')
    g.write('OFLAGS = '+self.setCompilers.getCompilerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.close()
    
    if self.installNeeded(mkfile):
      try:
        self.logPrintBox('Compiling chaco; this may take several minutes')
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';CHACO_INSTALL_DIR='+self.installDir+';export CHACO_INSTALL_DIR; cd code; make clean; make; cd '+self.installDir+'; '+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+' '+self.libdir+'/libchaco.'+self.setCompilers.AR_LIB_SUFFIX+' `find '+self.packageDir+'/code -name "*.o"`; cd '+self.libdir+'; '+self.setCompilers.AR+' d libchaco.'+self.setCompilers.AR_LIB_SUFFIX+' main.o; '+self.setCompilers.RANLIB+' libchaco.'+self.setCompilers.AR_LIB_SUFFIX, timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on CHACO: '+str(e))
      self.postInstall(output+err, mkfile)
    return self.installDir

