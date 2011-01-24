import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/blopex-1.1.1.tar.gz']
    self.functions  = ['lobpcg_solve']
    self.includes   = ['interpreter.h']
    self.liblist    = [['libBLOPEX.a']]
    self.complex    = 1
    self.requires32bitint = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    # Hypre has blopex sources/includes for eg: interpreter.h which can conflict with one in petsc.
    # hence attepmt to build hypre before blopex - so blopex's interpreter.h is the one that gets used.
    if self.framework.argDB.has_key('download-hypre') and self.framework.argDB['download-hypre']:
      if self.framework.argDB.has_key('download-openmpi') and self.framework.argDB['download-openmpi']:
        if self.framework.argDB.has_key('download-blopex') and self.framework.argDB['download-blopex']:
          raise RuntimeError('Cannot use BLOPEX with --download-hypre aswell as --download-openmpi.\n\
Suggest using --download-mpich or install openmpi separately - and specify mpicc etc to petsc configure.\n')
      self.hypre      = framework.require('PETSc.packages.hypre',self)
      self.deps       = [self.blasLapack,self.hypre]
    else:
      self.deps       = [self.blasLapack]
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'Makefile.inc'),'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC          = '+self.setCompilers.getCompiler()+'\n') 
    g.write('CFLAGS      = ' + self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','') +'\n')
    self.setCompilers.popLanguage()
    g.write('AR          = '+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB      = '+self.setCompilers.RANLIB+'\n')
    # blopex uses defaut 'make' targets, and this uses TARGET_ARCH var. If this var
    # is set incorrectly in user env - build breaks.
    g.write('TARGET_ARCH = \n')
    g.close()

    if self.installNeeded('Makefile.inc'):
      try:
        self.logPrintBox('Compiling blopex; this may take several minutes')
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';BLOPEX_INSTALL_DIR='+self.installDir+';export BLOPEX_INSTALL_DIR; make clean; make; mv -f lib/* '+os.path.join(self.installDir,self.libdir)+'; cp -fp multivector/temp_multivector.h include/.; mv -f include/* '+os.path.join(self.installDir,self.includedir)+'', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on BLOPEX: '+str(e))
      self.postInstall(output+err,'Makefile.inc')
    return self.installDir
