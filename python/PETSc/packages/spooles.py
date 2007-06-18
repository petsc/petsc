#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download   = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/spooles-2.2-June_2007.tar.gz']
    self.functions  = ['InpMtx_init']
    self.includes   = ['spoolesMPI.h']
    self.liblist    = [['spoolesMPI.a','spooles.a']]
    self.complex    = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
    # Get the SPOOLES directories
    spoolesDir = self.getDir()
    
    # Configure and Build SPOOLES
    g = open(os.path.join(spoolesDir,'Make.inc'),'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC          = '+self.setCompilers.getCompiler()+'\n') 
    g.write('CFLAGS      = ' + self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','') +'\n')
    self.setCompilers.popLanguage()
    g.write('AR          = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS     = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB      = '+self.setCompilers.RANLIB+'\n')
    g.write('MPI_LIBS    = '+self.libraries.toString(self.mpi.lib)+'\n') 
    g.write('MPI_INCLUDE_DIR = '+self.headers.toString(self.mpi.include)+'\n') 
    g.close()
    if not os.path.isfile(os.path.join(self.confDir,'spooles')) or not (self.getChecksum(os.path.join(self.confDir,'spooles')) == self.getChecksum(os.path.join(spoolesDir,'Make.inc'))):
      self.framework.log.write('Have to rebuild SPOOLES, Make.inc != '+self.confDir+'/spooles\n')
      try:
        self.logPrintBox('Compiling spooles; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+spoolesDir+'; SPOOLES_INSTALL_DIR='+self.installDir+'; export SPOOLES_INSTALL_DIR; make clean; make lib', timeout=2500, log = self.framework.log)[0]
        output  = config.base.Configure.executeShellCommand('cd '+spoolesDir+'; cp -f *.h '+self.installDir+'/include; HLISTS=`ls *.h`; for hlist in $HLISTS MPI.h; do dir=`echo ${hlist} | sed s/"\.h"//`; mkdir '+self.installDir+'/include/$dir; cp -f $dir/*.h '+self.installDir+'/include/$dir/.; done; mv -f *.a MPI/src/*.a '+self.installDir+'/lib', timeout=2500, log = self.framework.log)[0]        
      except RuntimeError, e:
        raise RuntimeError('Error running make on SPOOLES: '+str(e))
      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(spoolesDir,'Make.inc')+' '+self.confDir+'/spooles', timeout=5, log = self.framework.log)[0]
      #include "../
      self.checkInstall(output)
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed SPOOLES into '+self.installDir)
    else:
      self.framework.log.write('Do NOT need to compile SPOOLES downloaded libraries\n')  
    return self.installDir

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
