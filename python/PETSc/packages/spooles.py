#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download   = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/spooles-2.2.tar.gz']
    self.functions  = ['InpMtx_init']
    self.includes   = ['MPI/spoolesMPI.h']
    self.liblist    = [[os.path.join('MPI','src','spoolesMPI.a'),'spooles.a']]
    self.libdir     = ''
    self.includedir = ''
    self.complex    = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('PETSc.packages.MPI',self)
    self.blasLapack = framework.require('PETSc.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
    # Get the SPOOLES directories
    spoolesDir = self.getDir()
    installDir = os.path.join(spoolesDir, self.arch.arch)
    
    # Configure and Build SPOOLES
    if os.path.isfile(os.path.join(spoolesDir,'Make.inc')):
      output  = config.base.Configure.executeShellCommand('cd '+spoolesDir+'; rm -f Make.inc', timeout=2500, log = self.framework.log)[0]
    g = open(os.path.join(spoolesDir,'Make.inc'),'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC          = '+self.setCompilers.getCompiler()+'\n') 
    g.write('CFLAGS      = ' + self.setCompilers.getCompilerFlags() +'\n')
    self.setCompilers.popLanguage()
    g.write('AR          = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS     = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB      = '+self.setCompilers.RANLIB+'\n')
    g.write('MPI_LIBS    = '+self.libraries.toString(self.mpi.lib)+'\n') 
    g.write('MPI_INCLUDE_DIR = -I'+self.libraries.toString(self.mpi.include)+'\n') 
    g.close()
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(os.path.join(installDir,'Make.inc')) or not (self.getChecksum(os.path.join(installDir,'Make.inc')) == self.getChecksum(os.path.join(spoolesDir,'Make.inc'))):
      self.framework.log.write('Have to rebuild SPOOLES, Make.inc != '+installDir+'/Make.inc\n')
      try:
        self.logPrintBox('Compiling spooles; this may take several minutes')
        if os.path.isfile(os.path.join(installDir,'Make.inc')):
          output  = config.base.Configure.executeShellCommand('cd '+spoolesDir+'; rm -rf '+installDir, timeout=2500, log = self.framework.log)[0]
          os.mkdir(installDir)          
        output  = config.base.Configure.executeShellCommand('cd '+spoolesDir+'; SPOOLES_INSTALL_DIR='+installDir+'; export SPOOLES_INSTALL_DIR; make clean; make lib; HLISTS=`ls *.h`; cd '+self.arch.arch+'; for hlist in $HLISTS; do dir=`echo ${hlist} | sed s/"\.h"//`; mkdir $dir; cp ../$dir/*.h $dir/.; done; cp ../*.h .; cd ..; cd '+self.arch.arch+'; mv ../*.a .; mkdir MPI/src; mv ../MPI/src/*.a MPI/src/.', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SPOOLES: '+str(e))
      else:
        self.framework.log.write('Do NOT need to compile SPOOLES downloaded libraries\n')  
      if not os.path.isfile(os.path.join(installDir,self.libdir,'spooles.a')):
        self.framework.log.write('Error running make on SPOOLES   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on SPOOLES follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on SPOOLES *******\n')
        raise RuntimeError('Error running make on SPOOLES, libraries not installed')

      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(spoolesDir,'Make.inc')+' '+installDir, timeout=5, log = self.framework.log)[0]
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed SPOOLES into '+installDir)
    return self.getDir()

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
