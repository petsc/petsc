#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/SuperLU_3.1-Aug_3_2008.tar.gz']
    self.functions  = ['set_default_options']
    self.includes   = ['slu_ddefs.h']
    self.liblist    = [['libsuperlu_3.1.a']]
    self.complex    = 1
    # SuperLU has NO support for 64 bit integers, use SuperLU_Dist if you need that
    self.excludename = ['SuperLU_DIST']
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.blasLapack = self.framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return

  def Install(self):
    # Get the SUPERLU directories

    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    g.write('TMGLIB       = tmglib.a\n')
    g.write('SUPERLULIB   = libsuperlu_3.1.a\n')
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

    # set blas name mangling
    if self.blasLapack.mangling == 'underscore':
      g.write('CDEFS   = -DAdd_')
    elif self.blasLapack.mangling == 'caps':
      g.write('CDEFS   = -DUpCase')
    else:
      g.write('CDEFS   = -DNoChange')
    if self.framework.argDB['with-64-bit-indices']:
      g.write(' -D_LONGINT')
    g.write('\n')

    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      g.write('FORTRAN      = '+self.setCompilers.getCompiler()+'\n')
      g.write('FFLAGS       = '+self.setCompilers.getCompilerFlags().replace('-Mfree','')+'\n')
      self.setCompilers.popLanguage()
    else:
      g.write('FORTRAN    = \n')
      g.write('FFLAGS     = \n')
    g.write('MATLAB       =\n')
    g.write('NOOPTS       = '+self.blasLapack.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.blasLapack.getPrecisionFlag(self.setCompilers.getCompilerFlags())+' '+self.blasLapack.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())+'\n')
    g.close()
    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling superlu; this may take several minutes')
        output = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; SUPERLU_INSTALL_DIR='+self.installDir+'/lib; export SUPERLU_INSTALL_DIR; make clean; make lib LAAUX="" SLASRC="" DLASRC="" CLASRC="" ZLASRC="" SCLAUX="" DZLAUX=""; mv -f SRC/*.a '+os.path.join(self.installDir,'lib')+';  cp -f SRC/*.h '+os.path.join(self.installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUPERLU: '+str(e))
      self.checkInstall(output,'make.inc')
    return self.installDir

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by SuperLU'''
    '''Normally you do not need to provide this method'''
    PETSc.package.Package.configureLibrary(self) 
    if not self.blasLapack.checkForRoutine('slamch'): 
      raise RuntimeError('SuperLU requires the LAPACK routine slamch()')
    self.framework.log.write('Found slamch() in Lapack library as needed by SuperLU\n')

    if not self.blasLapack.checkForRoutine('dlamch'): 
      raise RuntimeError('SuperLU requires the LAPACK routine dlamch()')
    self.framework.log.write('Found dlamch() in Lapack library as needed by SuperLU\n')

    if not self.blasLapack.checkForRoutine('xerbla'): 
      raise RuntimeError('SuperLU requires the BLAS routine xerbla()')
    self.framework.log.write('Found xerbla() in BLAS library as needed by SuperLU\n')

    return
  
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
