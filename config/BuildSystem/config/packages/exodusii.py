import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'e7308f4401dd8692318d22bc6abb1dc84d3fb404'
    self.download          = ['git://https://github.com/gsjaardema/seacas.git','https://github.com/gsjaardema/seacas/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames  = ['seacas']
    self.functions         = ['ex_close']
    self.includes          = ['exodusII.h']
    self.liblist           = [['libexodus.a'], ]
    self.hastests          = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.pnetcdf = framework.require('config.packages.pnetcdf', self)
    self.netcdf  = framework.require('config.packages.netcdf', self)
    self.hdf5    = framework.require('config.packages.hdf5', self)
    self.deps = [self.hdf5,self.netcdf,self.pnetcdf]
    return

  def formCMakeConfigureArgs(self):
    import os
    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build exodusII\nSuggest adding --download-cmake to ./configure arguments')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    args.append('-DACCESSDIR:PATH='+self.installDir)
    args.append('-DCMAKE_INSTALL_PREFIX:PATH='+self.installDir)
    args.append('-DCMAKE_INSTALL_RPATH:PATH='+os.path.join(self.installDir,'lib'))
    self.pushLanguage('C')
    args.append('-DCMAKE_C_COMPILER:FILEPATH="'+self.getCompiler()+'"')
    self.popLanguage()
    # building the fortran library is technically not required to add exodus support
    # we build it anyway so that fortran users can still use exodus functions directly 
    # from their code
    if hasattr(self.setCompilers, 'FC'):
      self.pushLanguage('FC')
      args.append('-DCMAKE_Fortran_COMPILER:FILEPATH="'+self.getCompiler()+'"')
      args.append('-DSEACASProj_ENABLE_SEACASExodus_for=ON')
      args.append('-DSEACASProj_ENABLE_SEACASExoIIv2for32=ON')
      self.popLanguage()
    else:
      args.append('-DSEACASProj_ENABLE_SEACASExodus_for=OFF')
      args.append('-DSEACASProj_ENABLE_SEACASExoIIv2for32=OFF')
    args.append('-DSEACASProj_ENABLE_SEACASExodus=ON')
    args.append('-DSEACASProj_ENABLE_TESTS=ON')
    args.append('-DSEACASProj_SKIP_FORTRANCINTERFACE_VERIFY_TEST:BOOL=ON')
    args.append('-DTPL_ENABLE_Matio:BOOL=OFF')
    args.append('-DTPL_ENABLE_Netcdf:BOOL=ON')
    args.append('-DTPL_ENABLE_Pnetcdf:BOOL=ON')
    args.append('-DTPL_Netcdf_Enables_PNetcdf:BOOL=ON')
    args.append('-DTPL_ENABLE_MPI=ON')
    args.append('-DTPL_ENABLE_Pamgen=OFF')
    args.append('-DTPL_ENABLE_CGNS:BOOL=OFF')
    if not self.netcdf.directory:
      raise RuntimeError('NetCDF dir is not known! ExodusII requires explicit path to NetCDF. Suggest using --with-netcdf-dir or --download-netcdf')
    else:
      args.append('-DNetCDF_DIR:PATH='+self.netcdf.directory)
    args.append('-DHDF5_DIR:PATH='+self.hdf5.directory)
    if not self.pnetcdf.directory:
      raise RuntimeError('PNetCDF dir is not known! ExodusII requires explicit path to PNetCDF. Suggest using --with-pnetcdf-dir or --download-pnetcdf')
    else:
      args.append('-DPnetcdf_LIBRARY_DIRS:PATH='+os.path.join(self.pnetcdf.directory,'lib'))
      args.append('-DPnetcdf_INCLUDE_DIRS:PATH='+os.path.join(self.pnetcdf.directory,'include'))
    if self.checkSharedLibrariesEnabled():
      args.append('-DBUILD_SHARED_LIBS:BOOL=ON')
    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=Debug')
    else:
      args.append('-DCMAKE_BUILD_TYPE=Release')
    return args
