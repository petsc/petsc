#!/usr/bin/env python
import config.base
import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/exodusii-5.14-petsc.tgz']
    self.liblist    = [['libexoIIv2for.a', 'libexodus.a'], ['libexoIIv2for.a', 'libexoIIv2c.a']]
    self.functions  = ['ex_close']
    self.includes   = ['exodusII.h']
    self.includedir = ['include', os.path.join('cbind', 'include'), os.path.join('forbind', 'include')]
    self.altlibdir  = '.'
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.netcdf = framework.require('config.packages.netcdf', self)
    self.deps   = [self.netcdf]
    return

  def Install(self):
    self.framework.log.write('exodusIIDir = '+self.packageDir+' installDir '+self.installDir+'\n')

    mkfile = 'make.inc'
    g = open(os.path.join(self.packageDir, mkfile), 'w')
    self.framework.log.write(repr(dir(self.setCompilers)))
    self.setCompilers.pushLanguage('C')
    g.write('CC = '+self.setCompilers.getCompiler()+'\n')
    g.write('CC_FLAGS = '+self.setCompilers.getCompilerFlags()+'\n')
    self.setCompilers.popLanguage()

    self.setCompilers.pushLanguage('FC')
    g.write('FC = '+self.setCompilers.getCompiler()+'\n')
    g.write('FC_FLAGS = '+self.setCompilers.getCompilerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.write('RANLIB      = '+self.setCompilers.RANLIB+'\n')
    g.write('AR      = '+self.setCompilers.AR+'\n')
    g.write('AR_FLAGS      = '+self.setCompilers.AR_FLAGS+'\n')

    g.close()

    if self.installNeeded(mkfile):
      try:
        self.logPrintBox('Compiling ExodusII; this may take several minutes')
        output,err,ret = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && make -f Makefile.petsc clean && make -f Makefile.petsc && make -f Makefile.petsc install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on exodusII: '+str(e))
      self.postInstall(output+err, mkfile)
    return self.installDir
