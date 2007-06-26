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
    self.includes   = ['MPI/spoolesMPI.h']
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

    g = open(os.path.join(self.packageDir,'Make.inc'),'w')
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
    if self.installNeeded('Make.inc'):
      try:
        self.logPrintBox('Compiling spooles; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; SPOOLES_INSTALL_DIR='+self.installDir+'; export SPOOLES_INSTALL_DIR; make clean; make lib', timeout=2500, log = self.framework.log)[0]
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; cp -f *.h '+self.installDir+'/include; HLISTS=`ls *.h`; for hlist in $HLISTS MPI.h; do dir=`echo ${hlist} | sed s/"\.h"//`; mkdir '+self.installDir+'/include/$dir; cp -f $dir/*.h '+self.installDir+'/include/$dir/.; done; mv -f *.a MPI/src/*.a '+self.installDir+'/lib', timeout=2500, log = self.framework.log)[0]        
      except RuntimeError, e:
        raise RuntimeError('Error running make on SPOOLES: '+str(e))
      self.checkInstall(output,'Make.inc')
    return self.installDir

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
