#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import PETSc.package

import re
import os

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.mpi           = self.framework.require('PETSc.packages.MPI', self)
    self.blasLapack   = self.framework.require('PETSc.packages.BlasLapack',self)
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/spai_3.0.tar.gz']
    self.deps         = [self.mpi,self.blasLapack]
    self.functions    = ['bspai']
    self.includes     = ['spai.h']
    self.liblist      = ['libspai.a']
    # SPAI include files are in the lib directory
    self.includedir   = 'lib'
    return

  def Install(self):
    spaiDir = self.getDir()
    installDir = os.path.join(spaiDir, self.arch.arch)
    if not os.path.isdir(os.path.join(installDir,'lib')):
      os.mkdir(os.path.join(installDir,'lib'))      
    self.framework.pushLanguage('C')
    args = 'CC = '+self.framework.getCompiler()+'\nCFLAGS = -DMPI '+self.framework.getCompilerFlags()+' '+' '.join([self.libraries.getIncludeArgument(inc) for inc in self.mpi.include])+'\n'
    self.framework.popLanguage()
    try:
      fd      = file(os.path.join(installDir,'Makefile.in'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild Spai oldargs = '+oldargs+' new args '+args+'\n')
      self.framework.logClear()      
      self.logPrint('=================================================================================', debugSection='screen')
      self.logPrint("         Configuring and compiling Spai; this may take several minutes\n", debugSection='screen')
      self.logPrint('=================================================================================\n', debugSection='screen')
      fd = file(os.path.join(installDir,'Makefile.in'),'w')
      fd.write(args)
      fd.close()
      fd = file(os.path.join(spaiDir,'lib','Makefile.in'),'w')
      fd.write(args)
      fd.close()
      output  = config.base.Configure.executeShellCommand('cd '+os.path.join(spaiDir,'lib')+'; make ; mv libspai.a '+os.path.join(installDir,'lib','libspai.a'),timeout=250, log = self.framework.log)[0]
      output  = config.base.Configure.executeShellCommand('cd '+os.path.join(spaiDir,'lib')+'; cp *.h '+os.path.join(installDir,'lib'),timeout=250, log = self.framework.log)[0]      
      try:
        output  = config.base.Configure.executeShellCommand(self.setcompilers.RANLIB+' '+os.path.join(installDir,'lib')+'/libspai.a', timeout=250, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on SPAI libraries: '+str(e))
        
    return spaiDir


if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
