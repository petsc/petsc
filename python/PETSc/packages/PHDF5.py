#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['']
    self.functions = []
    self.includes  = ['hdf5.h']
    self.liblist   = [[]]
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.deps       = [self.mpi]  
    return

  def Install(self):

    self.framework.pushLanguage('C')
    args = ['--prefix='+self.installDir, 'CC="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"']
    self.framework.popLanguage()
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'phdf5'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('pdf5'):
      try:
        self.logPrintBox('Configuring parallel HDF5; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on HDF5: '+str(e))
      try:
        self.logPrintBox('Compiling parallel HDF5; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+'make ; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on HDF5: '+str(e))
      self.checkInstall(output,'hdf5')
    return self.installDir
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
