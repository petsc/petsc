#!/usr/bin/env python
import config.base
import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/exodus-5.24.tar.bz2']
    self.downloadfilename = 'exodus'
    self.liblist    = [['libexoIIv2for.a', 'libexodus.a'], ['libexoIIv2for.a', 'libexoIIv2c.a'], ['libexoIIv2c.a']]
    self.functions  = ['ex_close']
    self.includes   = ['exodusII.h']
    self.includedir = ['include']
    self.altlibdir  = '.'
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.netcdf = framework.require('config.packages.netcdf', self)
    # ExodusII does not call HDF5 directly, but it does call nc_def_var_deflate(), which is only
    # part of libnetcdf when built using --enable-netcdf-4.  Currently --download-netcdf (netcdf.py)
    # sets --enable-netcdf-4 only when HDF5 is enabled.
    self.hdf5   = framework.require('config.packages.hdf5', self)
    self.deps   = [self.netcdf, self.hdf5]
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
        builddir = os.path.join(self.packageDir, 'exodus')
        output,err,ret = config.base.Configure.executeShellCommand('cd '+builddir+' && make -f Makefile.standalone libexodus.a '+args, timeout=2500, log = self.framework.log)
        shutil.copy(os.path.join(builddir,'libexodus.a'),os.path.join(self.installDir,'lib'))
        for i in cincludes:
          shutil.copy(os.path.join(builddir,'cbind','include',i),os.path.join(self.installDir,'include'))
        if hasattr(self.setCompilers, 'FC'):
          output,err,ret = config.base.Configure.executeShellCommand('cd '+builddir+' && make -f Makefile.standalone libexoIIv2for.a '+args, timeout=2500, log = self.framework.log)
          shutil.copy(os.path.join(builddir,'libexoIIv2for.a'),os.path.join(self.installDir,'lib'))
          for i in fincludes:
            shutil.copy(os.path.join(builddir,'forbind','include',i),os.path.join(self.installDir,'include'))
      except RuntimeError, e:
        raise RuntimeError('Error running make on ExodusII: '+str(e))
      self.postInstall(output+err, mkfile)
    return self.installDir
