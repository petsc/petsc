#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    # use the version from PETSc ftp site - it has lapack removed
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/scalapack.tgz']
    self.includes  = []
    self.liblist   = [['libscalapack.a']]
    self.functions = ['pssytrd']
    self.functionsFortran = 1
    self.complex   = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.blacs      = framework.require('PETSc.packages.blacs',self)
    self.deps       = [self.blacs,self.mpi,self.blasLapack]
    return

  def Install(self):
    if not hasattr(self.setCompilers, 'FC'):
      raise RuntimeError('SCALAPACK requires Fortran for automatic installation')

    # Get the SCALAPACK directories
    scalapackDir = self.getDir()

    # Configure and build SCALAPACK
    g = open(os.path.join(scalapackDir,'SLmake.inc'),'w')
    g.write('SHELL        = /bin/sh\n')
    g.write('home         = '+self.getDir()+'\n')
    g.write('USEMPI       = -DUsingMpiBlacs\n')
    g.write('SENDIS       = -DSndIsLocBlk\n')
    if (self.mpi.commf2c):
      g.write('WHATMPI      = -DUseMpi2\n')
    else:
      g.write('WHATMPI      = -DCSAMEF77\n')
    g.write('BLACSDBGLVL  = -DBlacsDebugLvl=1\n')
    g.write('BLACSLIB     = '+self.libraries.toString(self.blacs.lib)+'\n') 
    g.write('SMPLIB       = '+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('SCALAPACKLIB = '+os.path.join(self.installDir,self.libdir,'libscalapack.a')+' \n')
    g.write('CBLACSLIB    = $(BLACSCINIT) $(BLACSLIB) $(BLACSCINIT)\n')
    g.write('FBLACSLIB    = $(BLACSFINIT) $(BLACSLIB) $(BLACSFINIT)\n')
    if self.compilers.fortranManglingDoubleUnderscore:
      blah = 'f77IsF2C'
    elif self.compilers.fortranMangling == 'underscore':
      blah = 'Add_'
    elif self.compilers.fortranMangling == 'capitalize':
      blah = 'UpCase'
    else:
      blah = 'NoChange'
    g.write('CDEFS        =-D'+blah+' -DUsingMpiBlacs\n')
    g.write('PBLASdir     = $(home)/PBLAS\n')
    g.write('SRCdir       = $(home)/SRC\n')
    g.write('TOOLSdir     = $(home)/TOOLS\n')
    g.write('REDISTdir    = $(home)/REDIST\n')
    self.setCompilers.pushLanguage('FC')  
    g.write('F77          = '+self.setCompilers.getCompiler()+'\n')
    g.write('F77FLAGS     = '+self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')+'\n')
    g.write('F77LOADER    = '+self.setCompilers.getLinker()+'\n')      
    g.write('F77LOADFLAGS = '+self.setCompilers.getLinkerFlags()+'\n')
    self.setCompilers.popLanguage()
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    g.write('CCFLAGS      = '+self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')+'\n')      
    g.write('CCLOADER     = '+self.setCompilers.getLinker()+'\n')
    g.write('CCLOADFLAGS  = '+self.setCompilers.getLinkerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.write('ARCH         = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')    
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')    
    g.close()

    if self.installNeeded('SLmake.inc'):
      try:
        output  = config.base.Configure.executeShellCommand('cd '+scalapackDir+';make cleanlib', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        pass
      try:
        self.logPrintBox('Compiling Scalapack; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+scalapackDir+';make', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SCALAPACK: '+str(e))
      self.checkInstall(output,'SLmake.inc')
    return self.installDir

  def checkLib(self,lib,func,mangle,otherLibs = []):
    oldLibs = self.compilers.LIBS
    found = self.libraries.check(lib,func, otherLibs = otherLibs+self.mpi.lib+self.blasLapack.lib+self.compilers.flibs,fortranMangle=mangle)
    self.compilers.LIBS=oldLibs
    if found:
      self.framework.log.write('Found function '+str(func)+' in '+str(lib)+'\n')
    return found

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by SCALAPACK'''
    '''Normally you do not need to provide this method'''
    # SCALAPACK requires ALL of BLAS/LAPACK
    if self.blasLapack.f2c:
      raise RuntimeError('SCALAPACK requires a COMPLETE BLAS and LAPACK, it cannot be used with the --download-c-blas-lapack\nUse --download-f-blas-lapack option instead.')
    PETSc.package.Package.configureLibrary(self)
    return
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
