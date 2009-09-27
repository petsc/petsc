import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/blopex_abstract_Aug_2006.tar.gz']
    self.functions  = ['lobpcg_solve']
    self.includes   = ['interpreter.h']
    self.liblist    = [['libBLOPEX.a']]
    self.complex    = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    if self.framework.argDB.has_key('download-hypre') and not self.framework.argDB['download-hypre'] == 0:
      self.hypre      = framework.require('PETSc.packages.hypre',self)
      self.deps       = [self.mpi,self.blasLapack,self.hypre]
    elif self.framework.argDB.has_key('with-hypre-dir') or self.framework.argDB.has_key('with-hypre-include') or self.framework.argDB.has_key('with-hypre-lib'):   
      self.hypre      = framework.require('PETSc.packages.hypre',self)
      self.deps       = [self.mpi,self.blasLapack,self.hypre]
    else:
      self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'Makefile.inc'),'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC          = '+self.setCompilers.getCompiler()+'\n') 
    g.write('CFLAGS      = ' + self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','') +'\n')
    self.setCompilers.popLanguage()
    g.write('AR          = '+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB      = '+self.setCompilers.RANLIB+'\n')
    g.close()

    if self.installNeeded('Makefile.inc'):
      try:
        self.logPrintBox('Compiling blopex; this may take several minutes')
        output  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';BLOPEX_INSTALL_DIR='+self.installDir+';export BLOPEX_INSTALL_DIR; make clean; make; mv -f lib/* '+os.path.join(self.installDir,self.libdir)+'; cp -fp multivector/temp_multivector.h include/.; mv -f include/* '+os.path.join(self.installDir,self.includedir)+'', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on BLOPEX: '+str(e))
      self.postInstall(output,'Makefile.inc')
    return self.installDir
