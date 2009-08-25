#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import shutil
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/PLAPACKR32-Aug-21-2009.tar.gz']
    self.functions  = ['PLA_LU']
    self.includes   = ['PLA.h']
    self.liblist    = [['libPLAPACK.a']]
    self.complex    = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
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
        output  = config.base.Configure.executeShellCommand('cp -f '+incDir+'/*.h '+installIncDir, timeout=2500, log = self.framework.log)[0]        
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';make removeall; make', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on PLAPACK: '+str(e))
      self.postInstall(output,'Make.include')
    return self.installDir

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by PLAPACK'''
    '''Normally you do not need to provide this method'''
    if self.blasLapack.f2c:
      raise RuntimeError('PLAPACK requires a COMPLETE BLAS and LAPACK, it cannot be used with --download-c-blas-lapack=1 \nUse --download-f-blas-lapack option instead.')

    PETSc.package.Package.configureLibrary(self)
    # PLAPACK requires BLAS complex and single routines()
    if not self.blasLapack.checkForRoutine('sscal') or not self.blasLapack.checkForRoutine('cscal'):
      raise RuntimeError('PLAPACK requires the complex and single precision BLAS routines, the current BLAS libraries '+str(self.blasLapack.lib)+' does not have it\nYou need a COMPLETE install of BLAS: --download-f-blas-lapack is NOT a complete BLAS library')
    self.framework.log.write('Found sscal() and cscal() in BLAS library as needed by PLAPACK\n')
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
