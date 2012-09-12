import config.package

# Eventually, we should support HDF5:
#   ./configure --enable-netcdf-4 --with-hdf5=/home/ed/local --with-zlib=/home/ed/local --prefix=/home/ed/local

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.downloadpath    = 'http://www.unidata.ucar.edu/downloads/netcdf/ftp/'
    self.downloadext     = 'tar.gz'
    self.downloadversion = '4.2.1.1'
    self.functions       = ['nccreate']
    self.includes        = ['netcdf.h']
    self.liblist         = [['libnetcdf.a']]
    self.cxx             = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi   = framework.require('config.packages.MPI', self)
    self.hdf5  = framework.require('config.packages.hdf5', self)
    self.odeps = [self.mpi, self.hdf5]
    self.deps  = [self.mpi]
    return

  def Install(self):
    import os, sys

    configOpts     = []
    # Unused flags: F90, CPPFLAGS, LIBS, FLIBS
    configOpts.append('AR="'+self.setCompilers.AR+'"')
    configOpts.append('ARFLAGS="'+self.setCompilers.AR_FLAGS+'"')

    configOpts.append('--prefix='+self.installDir)
    configOpts.append('--libdir='+os.path.join(self.installDir,self.libdir))
    configOpts.append('--disable-dap')
    configOpts.append('--disable-hdf4')
    configOpts.append('--disable-netcdf-4')

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

    if hasattr(self.setCompilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      fcflags = self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')
      fcflags += ' ' + self.headers.toString(self.mpi.include)+' '+self.headers.toString('.')
      configOpts.append('FC="'+self.setCompilers.getCompiler()+'"')
      configOpts.append('FCFLAGS="'+fcflags+'"')
      if self.compilers.fortranIsF90:
        configOpts.append('F90="'+self.setCompilers.getCompiler()+'"')
      else:
        configOpts.append('--disable-f90')
      self.setCompilers.popLanguage()
    else:
      configOpts.append('--disable-f77')

    if self.setCompilers.sharedLibraries:
      configOpts.append('--enable-shared')

    args = ' '.join(configOpts)
    fd = file(os.path.join(self.packageDir,'netcdf'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('netcdf'):
      try:
        self.logPrintBox('Configuring NetCDF; this may take several minutes')
        output,err,ret  = self.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=2500, log = self.framework.log)
        self.logPrintBox('Compiling & installing NetCDF; this may take several minutes')
        output,err,ret  = self.executeShellCommand('cd '+self.packageDir+' && make clean && make && make install && make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on NetCDF: '+str(e))
      self.postInstall(output+err,'netcdf')
    return self.installDir
