#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download   = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/SuperLU_3.0-Jan_5_2006.tar.gz']
    self.functions  = ['set_default_options']
    self.includes   = ['slu_ddefs.h']
    self.libdir     = ''
    self.includedir = 'SRC'
    self.liblist    = [['libsuperlu_3.0.a']]
    self.complex    = 1
    self.excludename = ['SuperLU_DIST']
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.blasLapack = self.framework.require('PETSc.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return

  def Install(self):
    # Get the SUPERLU directories
    superluDir = self.getDir()
    installDir = os.path.join(superluDir, self.arch.arch)

    # Configure and Build SUPERLU
    if os.path.isfile(os.path.join(superluDir,'make.inc')):
      output  = config.base.Configure.executeShellCommand('cd '+superluDir+'; rm -f make.inc', timeout=2500, log = self.framework.log)[0]
    g = open(os.path.join(superluDir,'make.inc'),'w')
    g.write('TMGLIB       = tmglib.a\n')
    g.write('SUPERLULIB   = libsuperlu_3.0.a\n')
    g.write('BLASLIB      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('BLASDEF      = -DUSE_VENDOR_BLAS\n')
    g.write('ARCH         = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS       = '+self.setCompilers.getCompilerFlags()+'\n')
    g.write('LOADER       = '+self.setCompilers.getLinker()+'\n') 
    g.write('LOADOPTS     = \n') 
    self.setCompilers.popLanguage()
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      g.write('FORTRAN      = '+self.setCompilers.getCompiler()+'\n')
      g.write('FFLAGS       = '+self.setCompilers.getCompilerFlags().replace('-Mfree','')+'\n')
      # set fortran name mangling
      if self.compilers.fortranMangling == 'underscore':
        g.write('CDEFS   = -DAdd_\n')
      elif self.compilers.fortranMangling == 'capitalize':
        g.write('CDEFS   = -DUpCase\n')
      else:
        g.write('CDEFS   = -DNoChange\n')
      self.setCompilers.popLanguage()
    else:
      g.write('FORTRAN    = \n')
      g.write('FFLAGS     = \n')
    g.write('MATLAB       =\n')
    g.write('NOOPTS       =  -O0\n')
    g.close()
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(os.path.join(installDir,'make.inc')) or not (self.getChecksum(os.path.join(installDir,'make.inc')) == self.getChecksum(os.path.join(superluDir,'make.inc'))):
      self.framework.log.write('Have to rebuild SuperLU, make.inc != '+installDir+'/make.inc\n')
      try:
        self.logPrintBox('Compiling superlu; this may take several minutes')
        output = config.base.Configure.executeShellCommand('cd '+superluDir+'; SUPERLU_INSTALL_DIR='+installDir+'; export SUPERLU_INSTALL_DIR; make clean; make lib LAAUX="" SLASRC="" DLASRC="" CLASRC="" ZLASRC="" SCLAUX="" DZLAUX=""; mv *.a '+os.path.join(installDir,self.libdir)+'; mkdir '+os.path.join(installDir,self.includedir)+'; cp SRC/*.h '+os.path.join(installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUPERLU: '+str(e))
    else:
      self.framework.log.write('Do NOT need to compile SuperLU downloaded libraries\n')  
    if not os.path.isfile(os.path.join(installDir,self.libdir,'libsuperlu_3.0.a')):
        self.framework.log.write('Error running make on SUPERLU   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on SUPERLU follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on SUPERLU *******\n')
        raise RuntimeError('Error running make on SUPERLU, libraries not installed')
      
    output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(superluDir,'make.inc')+' '+installDir, timeout=5, log = self.framework.log)[0]
    self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed SUPERLU into '+installDir)
    return self.getDir()

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by SuperLU'''
    '''Normally you do not need to provide this method'''
    PETSc.package.Package.configureLibrary(self) 
    if self.blasLapack.f2c:
      raise RuntimeError('SuperLU requires a COMPLETE BLAS and LAPACK, it cannot be used with or --download-c-blas-lapack=1 \nUse --download-f-blas-lapack option instead.')

    # SuperLU requires slamch() & dlamch() LAPACK routines and PETSc version of superlu
    # have the internal versions disabled in favour of generic blas/lapack
    if not self.blasLapack.checkForRoutine('slamch'): 
      raise RuntimeError('SuperLU requires the LAPACK routine slamch(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it\nTry using --download-f-blas-lapack=1 option \nIf you are using the IBM ESSL library, it does not contain this function')
    self.framework.log.write('Found slamch() in Lapack library as needed by SuperLU\n')

    if not self.blasLapack.checkForRoutine('dlamch'): 
      raise RuntimeError('SuperLU requires the LAPACK routine dlamch(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it\nTry using --download-f-blas-lapack=1 option \nIf you are using the IBM ESSL library, it does not contain this function.')
    self.framework.log.write('Found dlamch() in Lapack library as needed by SuperLU\n')
    return
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
