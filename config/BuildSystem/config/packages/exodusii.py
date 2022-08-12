import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'v2022-08-01'
    self.download          = ['git://https://github.com/gsjaardema/seacas.git','https://github.com/gsjaardema/seacas/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames  = ['seacas']
    self.functions         = ['ex_close']
    self.includes          = ['exodusII.h']
    self.liblist           = [['libexodus.a'], ]
    return

  def setupHelp(self, help):
    config.package.CMakePackage.setupHelp(self,help)
    import nargs
    # PETSc does not need the Fortran interface. However some fortran examples test with it - so its enabled by default
    help.addArgument('EXODUSII', '-with-exodusii-fortran-bindings', nargs.ArgBool(None, 1, 'Use/build exodusii Fortran interface (PETSc does not need it)'))

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.pnetcdf = framework.require('config.packages.pnetcdf', self)
    self.netcdf  = framework.require('config.packages.netcdf', self)
    self.hdf5    = framework.require('config.packages.hdf5', self)
    self.deps = [self.hdf5,self.netcdf,self.pnetcdf]
    return

  def formCMakeConfigureArgs(self):
    import os
    import sys
    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build exodusII\nSuggest adding --download-cmake to ./configure arguments')
    if 'with-exodusii-fortran-bindings' in self.framework.clArgDB and self.argDB['with-exodusii-fortran-bindings'] and not hasattr(self.setCompilers, 'FC'):
      raise RuntimeError('exodusii fortran bindings requested but no fortran compiler detected')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    args.append('-DPYTHON_EXECUTABLE:PATH='+sys.executable)
    args.append('-DPythonInterp_FIND_VERSION:STRING={0}.{1}'.format(sys.version_info[0],sys.version_info[1]))
    args.append('-DACCESSDIR:PATH='+self.installDir)
    args.append('-DCMAKE_INSTALL_RPATH:PATH='+os.path.join(self.installDir,'lib'))
    args.append('-DSeacas_ENABLE_SEACASExodus:BOOL=ON')
    if self.argDB['with-exodusii-fortran-bindings'] and hasattr(self.setCompilers, 'FC'):
      args.append('-DSeacas_ENABLE_Fortran:BOOL=ON')
      args.append('-DSeacas_ENABLE_SEACASExoIIv2for32:BOOL=ON')
      args.append('-DSeacas_ENABLE_SEACASExoIIv2for:BOOL=ON')
      args.append('-DSeacas_ENABLE_SEACASExodus_for:BOOL=ON')
      args.append('-DSEACASProj_SKIP_FORTRANCINTERFACE_VERIFY_TEST:BOOL=ON')
    else:
      args.append('-DSeacas_ENABLE_Fortran:BOOL=OFF')
      args.append('-DSeacas_ENABLE_SEACASExoIIv2for32:BOOL=OFF')
      args.append('-DSeacas_ENABLE_SEACASExoIIv2for:BOOL=OFF')
      args.append('-DSeacas_ENABLE_SEACASExodus_for:BOOL=OFF')
    # exodiff requires a whole new set of dependencies so it is now disabled
    args.append('-DSeacas_ENABLE_SEACASExodiff:BOOL=OFF')
    args.append('-DSeacas_ENABLE_SEACASExotxt:BOOL=OFF')
    args.append('-DTPL_ENABLE_Matio:BOOL=OFF')
    args.append('-DTPL_ENABLE_Netcdf:BOOL=ON')
    args.append('-DTPL_ENABLE_Pnetcdf:BOOL=ON')
    args.append('-DTPL_Netcdf_Enables_PNetcdf:BOOL=ON')
    args.append('-DTPL_ENABLE_MPI:BOOL=ON')
    args.append('-DTPL_ENABLE_Pamgen:BOOL=OFF')
    args.append('-DTPL_ENABLE_CGNS:BOOL=OFF')
    args.append('-DTPL_ENABLE_fmt=OFF')
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
      args.append('-DSEACASExodus_ENABLE_SHARED:BOOL=ON')
      args.append('-DCMAKE_SHARED_LINKER_FLAGS:STRING="'+self.libraries.toString(self.dlib)+' '+self.compilers.LIBS+'"')
    return args

  def generateLibList(self, framework):
    ''' '''
    if self.argDB['with-exodusii-fortran-bindings'] and hasattr(self.setCompilers, 'FC'):
      self.liblist = [['libexoIIv2for32.a',] + libs for libs in self.liblist] + self.liblist
    return config.package.Package.generateLibList(self, framework)

