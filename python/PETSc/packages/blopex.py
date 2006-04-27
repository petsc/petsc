#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download   = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/blopex_abstract_Nov_2005.tar.gz']
    self.functions  = ['lobpcg_solve']
    self.includes   = ['interpreter.h']
    self.liblist    = [['libBLOPEX.a']]
    self.libdir     = 'lib'
    self.includedir = 'include'
    self.complex    = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
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
    # Get the BLOPEX directories
    blopexDir = self.getDir()
    installDir = os.path.join(blopexDir, self.arch.arch)
    
    # Configure and Build BLOPEX
    if os.path.isfile(os.path.join(blopexDir,'Makefile.inc')):
      output  = config.base.Configure.executeShellCommand('cd '+blopexDir+'; rm -f Makefile.inc', timeout=2500, log = self.framework.log)[0]
    g = open(os.path.join(blopexDir,'Makefile.inc'),'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC          = '+self.setCompilers.getCompiler()+'\n') 
    g.write('CFLAGS      = ' + self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','') +'\n')
    self.setCompilers.popLanguage()
    g.write('AR          = '+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB      = '+self.setCompilers.RANLIB+'\n')
    g.close()
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(os.path.join(installDir,'Makefile.inc')) or not (self.getChecksum(os.path.join(installDir,'Makefile.inc')) == self.getChecksum(os.path.join(blopexDir,'Makefile.inc'))):
      self.framework.log.write('Have to rebuild BLOPEX, Makefile.inc != '+installDir+'/Makefile.inc\n')
      try:
        self.logPrintBox('Compiling blopex; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+blopexDir+';BLOPEX_INSTALL_DIR='+installDir+';export BLOPEX_INSTALL_DIR; make clean; rm -f '+os.path.join(installDir,self.libdir)+'; rm -f '+os.path.join(installDir,self.includedir)+'; make; mv lib '+os.path.join(installDir,self.libdir)+'; cp -fp multivector/temp_multivector.h include/.; mv include '+os.path.join(installDir,self.includedir)+'', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on BLOPEX: '+str(e))
      else:
        self.framework.log.write('Do NOT need to compile BLOPEX downloaded libraries\n')  
      if not os.path.isfile(os.path.join(installDir,self.libdir,'libBLOPEX.a')):
        self.framework.log.write('Error running make on BLOPEX   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on BLOPEX follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on BLOPEX *******\n')
        raise RuntimeError('Error running make on BLOPEX, libraries not installed')

      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(blopexDir,'Makefile.inc')+' '+installDir, timeout=5, log = self.framework.log)[0]
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed BLOPEX into '+installDir)
    return self.getDir()

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
