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
    self.download   = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/PLAPACKR32.tar.gz']
    self.functions  = ['PLA_LU']
    self.includes   = ['PLA.h']
    self.libdir     = ''
    self.liblist    = [['libPLAPACK.a']]
    self.includedir = 'INCLUDE'
    self.complex    = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('PETSc.packages.MPI',self)
    self.blasLapack = framework.require('PETSc.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
    # Get the PLAPACK directories
    plapackDir = self.getDir()
    installDir = os.path.join(plapackDir, self.arch.arch)
    # Configure and Build PLAPACK
    plapackMakefile        = os.path.join(plapackDir,'Make.include')
    plapackInstallMakefile = os.path.join(installDir,'Make.include')
    if os.path.isfile(plapackMakefile): os.remove(plapackMakefile)
    g = open(plapackMakefile,'w')
    g.write('PLAPACK_ROOT = '+installDir+'\n')
    g.write('MANUFACTURE  = 50\n')  #PC
    g.write('MACHINE_TYPE = 500\n')  #LINUX
    g.write('BLASLIB      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('MPILIB       = '+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('MPI_INCLUDE  = -I'+self.libraries.toString(self.mpi.include)+'\n') 
    g.write('LIB          = $(BLASLIB) $(MPILIB)\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n') 
    g.write('CFLAGS       = -I$(PLAPACK_ROOT)/INCLUDE $(MPI_INCLUDE) -DMACHINE_TYPE=$(MACHINE_TYPE) -DMANUFACTURE=$(MANUFACTURE) ' + self.setCompilers.getCompilerFlags() +'\n')
    self.setCompilers.popLanguage()
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      g.write('FC           = '+self.setCompilers.getCompiler()+'\n')
      g.write('FFLAGS       = '+self.setCompilers.getCompilerFlags()+'\n')
      self.setCompilers.popLanguage()
    g.write('LINKER       = $(CC)\n')     #required by PLAPACK's examples
    g.write('LFLAGS       = $(CFLAGS)\n') #required by PLAPACK's examples
    g.write('AR           = '+self.setCompilers.AR+'\n')
    g.write('SED          = sed\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    g.write('PLAPACKLIB   =  $(PLAPACK_ROOT)/libPLAPACK.a\n')
    g.close()
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(plapackInstallMakefile) or not (self.getChecksum(plapackInstallMakefile) == self.getChecksum(plapackMakefile)):  
      self.framework.log.write('Have to rebuild PLAPACK, Make.include != '+plapackInstallMakefile+'\n')
      try:
        self.logPrintBox('Compiling PLAPACK; this may take several minutes')
        incDir = os.path.join(plapackDir,self.includedir)
        installIncDir = os.path.join(installDir,self.includedir)
        if os.path.isdir(installIncDir): shutil.rmtree(installIncDir)
        shutil.copytree(incDir,installIncDir);
        output  = config.base.Configure.executeShellCommand('cd '+plapackDir+';make removeall; make', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on PLAPACK: '+str(e))
      if not os.path.isfile(os.path.join(installDir,'libPLAPACK.a')):
        self.framework.log.write('Error running make on PLAPACK   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on PLAPACK follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on PLAPACK *******\n')
        raise RuntimeError('Error running make on PLAPACK, libraries not installed')
      
      output  = shutil.copy(plapackMakefile,plapackInstallMakefile)
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed PLAPACK into '+installDir)
    return self.getDir()

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
