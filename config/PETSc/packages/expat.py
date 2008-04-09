#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
#    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/expat-2.0.0.tar.gz']
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/expat-1.95.8.tar.gz']
    self.functions = ['XML_ExpatVersion']
    self.liblist   = [['libexpat.a']]
    self.includes  = ['expat.h']
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.compilers = framework.require('config.compilers', self)
    return

  def Install(self):

    self.framework.pushLanguage('C')
    flags = self.framework.getCompilerFlags()
    #  expat autoconf turns on GCC options if it thinks you are using a GNU compile :-(
    if config.setCompilers.Configure.isIntel(self.framework.getCompiler()):
      flags = flags + ' -gcc-sys'
    args = ['--prefix='+self.installDir, 'CC="'+self.framework.getCompiler()+' '+flags+'"']
    self.framework.popLanguage()
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'expat'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('expat'):
      try:
        self.logPrintBox('Configuring expat; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on expat: '+str(e))
      try:
        self.logPrintBox('Compiling expat; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; make ; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on expat: '+str(e))
      #need to run ranlib on the libraries using the full path
      try:
        output  = config.base.Configure.executeShellCommand(self.setCompilers.RANLIB+' '+os.path.join(self.installDir,'lib')+'/lib*.a', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on expat libraries: '+str(e))
      self.checkInstall(output,'expat')
    return self.installDir


if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
