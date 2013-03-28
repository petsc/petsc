#!/usr/bin/env python
import config.base
import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/exodusii-5.22b.tar.gz']
    self.liblist    = [['libexoIIv2for.a', 'libexodus.a'], ['libexoIIv2for.a', 'libexoIIv2c.a'], ['libexoIIv2c.a']]
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
    self.logPrintBox('Compiling ExodusII; this may take several minutes')
    import os
    import shutil
    configOpts     = []
    configOpts.append('RANLIB="'+self.setCompilers.RANLIB+'"')
    configOpts.append('AR="'+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'"')
    configOpts.append('NETCDF="'+self.installDir+'"')

    self.setCompilers.pushLanguage('C')
    configOpts.append('CC="'+self.setCompilers.getCompiler()+'"')
    configOpts.append('CCOPTIONS="'+self.setCompilers.getCompilerFlags()+' -DADDC_ "')
    self.setCompilers.popLanguage()

    if hasattr(self.setCompilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      configOpts.append('FC="'+self.setCompilers.getCompiler()+'"')
      configOpts.append('F77OPTIONS="'+self.setCompilers.getCompilerFlags()+'"')
      self.setCompilers.popLanguage()

    mkfile = 'make.inc'
    g = open(os.path.join(self.packageDir, mkfile), 'w')
    self.framework.log.write(repr(dir(self.setCompilers)))

    args = ' '.join(configOpts)
    fd = file(os.path.join(self.packageDir,'exodusii'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('exodusii'):
      cincludes  = ['exodusII.h','exodusII_cfg.h','exodusII_int.h','exodusII_par.h']
      fincludes  = ['exodusII.inc','exodusII_int.inc']
      try:
        self.logPrintBox('Compiling ExodusII; this may take several minutes')
        output,err,ret = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && make -f Makefile.standalone libexodus.a '+args, timeout=2500, log = self.framework.log)
        shutil.copy(os.path.join(self.packageDir,'libexodus.a'),os.path.join(self.installDir,'lib'))
        for i in cincludes:
          shutil.copy(os.path.join(self.packageDir,'cbind','include',i),os.path.join(self.installDir,'include'))
        if hasattr(self.setCompilers, 'FC'):
          output,err,ret = config.base.Configure.executeShellCommand('cd '+self.packageDir+' && make -f Makefile.standalone libexoIIv2for.a '+args, timeout=2500, log = self.framework.log)
          shutil.copy(os.path.join(self.packageDir,'libexoIIv2for.a'),os.path.join(self.installDir,'lib'))
          for i in fincludes:
            shutil.copy(os.path.join(self.packageDir,'forbind','include',i),os.path.join(self.installDir,'include'))
      except RuntimeError, e:
        raise RuntimeError('Error running make on exodusII: '+str(e))
      self.postInstall(output+err, mkfile)
    return self.installDir
