import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/PLAPACKR32-hg.tar.gz']
    self.functions  = ['PLA_LU']
    self.includes   = ['PLA.h']
    self.liblist    = [['libPLAPACK.a']]
    self.complex    = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
    import os
    incDir                 = os.path.join(self.packageDir,'INCLUDE')
    installIncDir          = os.path.join(self.installDir,self.includedir)
    plapackMakefile        = os.path.join(self.packageDir,'Make.include')
    plapackInstallMakefile = os.path.join(self.confDir,'PLAPACK')
    g = open(plapackMakefile,'w')
    g.write('PLAPACK_ROOT = '+self.installDir+'\n')
    if self.blasLapack.mangling == 'underscore':
      g.write('MANUFACTURE  = 50\n')  #PC
      g.write('MACHINE_TYPE = 500\n') #LINUX
    else:
      g.write('MANUFACTURE  = 20\n')  #IBM
      g.write('MACHINE_TYPE = 500\n') #SP2
    g.write('BLASLIB      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('MPILIB       = '+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('MPI_INCLUDE  = '+self.headers.toString(self.mpi.include)+'\n') 
    g.write('LIB          = $(BLASLIB) $(MPILIB)\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n') 
    g.write('CFLAGS       = -I'+installIncDir+' $(MPI_INCLUDE) -DMACHINE_TYPE=$(MACHINE_TYPE) -DMANUFACTURE=$(MANUFACTURE) '+self.setCompilers.getCompilerFlags()+'\n')
    self.setCompilers.popLanguage()
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      g.write('FC           = '+self.setCompilers.getCompiler()+'\n')
      g.write('FFLAGS       = '+self.setCompilers.getCompilerFlags()+'\n')
      self.setCompilers.popLanguage()
    else:
      raise RuntimeError('PLAPACK requires a fortran compiler! No fortran compiler configured!')
    g.write('LINKER       = $(CC)\n')     #required by PLAPACK's examples
    g.write('LFLAGS       = $(CFLAGS)\n') #required by PLAPACK's examples
    g.write('AR           = '+self.setCompilers.AR+'\n')
    g.write('SED          = sed\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    g.write('PLAPACKLIB   =  $(PLAPACK_ROOT)/lib/libPLAPACK.a\n')
    g.close()
    if self.installNeeded('Make.include'):
      try:
        self.logPrintBox('Compiling PLAPACK; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cp -f '+incDir+'/*.h '+installIncDir, timeout=2500, log = self.framework.log)        
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make removeall && make', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on PLAPACK: '+str(e))
      self.postInstall(output1+err1+output2+err2,'Make.include')
    return self.installDir

  def consistencyChecks(self):
    PETSc.package.NewPackage.consistencyChecks(self)
    if self.framework.argDB['with-'+self.package]:
      if self.blasLapack.f2c:
        raise RuntimeError('PLAPACK requires a COMPLETE BLAS and LAPACK, it cannot be used with --download-f2cblaslapack=1 \nUse --download-f-blas-lapack option instead.')
      if not self.blasLapack.checkForRoutine('sscal') or not self.blasLapack.checkForRoutine('cscal'):
        raise RuntimeError('PLAPACK requires the complex and single precision BLAS routines, the current BLAS libraries '+str(self.blasLapack.lib)+' does not have it\nYou need a COMPLETE install of BLAS: --download-f-blas-lapack is NOT a complete BLAS library')
      self.framework.log.write('Found sscal() and cscal() in BLAS library as needed by PLAPACK\n')
    return
