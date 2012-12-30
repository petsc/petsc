import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.downloadpath     = 'http://www.unidata.ucar.edu/downloads/netcdf/ftp/'
    self.downloadext      = 'tar.gz'
    self.downloadversion  = '4-4.2'
    self.functions        = []
    self.includes         = ['netcdfcpp.h']
    self.liblist          = [['libnetcdf_c++.a']]
    self.cxx              = 1
    return

  def setupDownload(self):
    '''Need this because the default puts a '-' between the name and the version number'''
    self.download = [self.downloadpath+self.downloadname+self.downloadversion+'.'+self.downloadext]

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi    = framework.require('config.packages.MPI', self)
    self.hdf5   = framework.require('config.packages.hdf5', self)
    self.netcdf = framework.require('config.packages.netcdf', self)
    self.odeps  = [self.mpi, self.hdf5, self.netcdf]
    self.deps   = [self.mpi]
    return

  def Install(self):
    import os, sys

    configOpts     = []
    # Unused flags: F90, CPPFLAGS, LIBS, FLIBS
    configOpts.append('AR="'+self.setCompilers.AR+'"')
    configOpts.append('ARFLAGS="'+self.setCompilers.AR_FLAGS+'"')

    configOpts.append('--prefix='+self.installDir)
    configOpts.append('--libdir='+os.path.join(self.installDir,self.libdir))

    self.setCompilers.pushLanguage('C')
    cflags = self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')
    cflags += ' ' + self.headers.toString(self.mpi.include)+' '+self.headers.toString('.')
    configOpts.append('CC="'+self.setCompilers.getCompiler()+'"')
    configOpts.append('CFLAGS="'+cflags+'"')
    self.setCompilers.popLanguage()

    if hasattr(self.setCompilers, 'CXX'):
      self.setCompilers.pushLanguage('Cxx')
      cxxflags = self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')
      cxxflags += ' ' + self.headers.toString(self.mpi.include)+' '+self.headers.toString('.')
      configOpts.append('CXX="'+self.setCompilers.getCompiler()+'"')
      configOpts.append('CXXFLAGS="'+cxxflags+'"')
      self.setCompilers.popLanguage()
    else:
      configOpts.append('--disable-cxx')

    if self.setCompilers.sharedLibraries:
      configOpts.append('--enable-shared')

    args = ' '.join(configOpts)
    fd = file(os.path.join(self.packageDir,'netcdfcxx'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('netcdfcxx'):
      try:
        self.logPrintBox('Configuring NetCDF-C++; this may take several minutes')
        output,err,ret  = self.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=2500, log = self.framework.log)
        self.logPrintBox('Compiling & installing NetCDF-C++; this may take several minutes')
        output,err,ret  = self.executeShellCommand('cd '+self.packageDir+' && make clean && make && make install && make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on NetCDF-C++: '+str(e))
      self.postInstall(output+err,'netcdfcxx')
    return self.installDir
