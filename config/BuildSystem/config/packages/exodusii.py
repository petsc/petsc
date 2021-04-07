import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'v2021-01-20'
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
    import sys
    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build exodusII\nSuggest adding --download-cmake to ./configure arguments')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    args.append('-DPYTHON_EXECUTABLE:PATH='+sys.executable)
    args.append('-DPythonInterp_FIND_VERSION:STRING={0}.{1}'.format(sys.version_info[0],sys.version_info[1]))
    args.append('-DACCESSDIR:PATH='+self.installDir)
    args.append('-DCMAKE_INSTALL_RPATH:PATH='+os.path.join(self.installDir,'lib'))
    # building the fortran library is technically not required to add exodus support
    # we build it anyway so that fortran users can still use exodus functions directly
    # from their code
    if hasattr(self.setCompilers, 'FC'):
      self.pushLanguage('FC')
      args.append('-DSEACASProj_ENABLE_SEACASExodus_for:BOOL=ON')
      args.append('-DSEACASProj_ENABLE_SEACASExoIIv2for32:BOOL=ON')
      args.append('-DSEACASExodus_for_ENABLE_TESTS:BOOL=OFF')
      self.popLanguage()
    else:
      args.append('-DSEACASProj_ENABLE_SEACASExodus_for:BOOL=OFF')
      args.append('-DSEACASProj_ENABLE_SEACASExoIIv2for32:BOOL=OFF')
    args.append('-DSEACASProj_ENABLE_SEACASExodus:BOOL=ON')
    # exodiff and exotxt are convenient tools to debug exodusII functionalities
    args.append('-DSEACASProj_ENABLE_SEACASExodif:BOOL=ON')
    if hasattr(self.setCompilers, 'FC'):
      args.append('-DSEACASProj_ENABLE_SEACASExotxt:BOOL=ON')
    else:
      args.append('-DSEACASProj_ENABLE_SEACASExotxt:BOOL=OFF')
    args.append('-DSEACASProj_ENABLE_TESTS:BOOL=OFF')
    args.append('-DSEACASProj_SKIP_FORTRANCINTERFACE_VERIFY_TEST:BOOL=ON')
    args.append('-DTPL_ENABLE_Matio:BOOL=OFF')
    args.append('-DTPL_ENABLE_Netcdf:BOOL=ON')
    args.append('-DTPL_ENABLE_Pnetcdf:BOOL=ON')
    args.append('-DTPL_Netcdf_Enables_PNetcdf:BOOL=ON')
    args.append('-DTPL_ENABLE_MPI:BOOL=ON')
    args.append('-DTPL_ENABLE_Pamgen:BOOL=OFF')
    args.append('-DTPL_ENABLE_CGNS:BOOL=OFF')
    if not self.netcdf.directory:
      raise RuntimeError('NetCDF dir is not known! ExodusII requires explicit path to NetCDF. Suggest using --with-netcdf-dir or --download-netcdf')
    else:
      args.append('-DNetCDF_DIR:PATH='+self.netcdf.directory)
    args.append('-DHDF5_ROOT:PATH='+self.hdf5.directory)
    if not self.pnetcdf.directory:
      raise RuntimeError('PNetCDF dir is not known! ExodusII requires explicit path to PNetCDF. Suggest using --with-pnetcdf-dir or --download-pnetcdf')
    else:
      args.append('-DPnetcdf_LIBRARY_DIRS:PATH='+os.path.join(self.pnetcdf.directory,'lib'))
      args.append('-DPnetcdf_INCLUDE_DIRS:PATH='+os.path.join(self.pnetcdf.directory,'include'))
    if self.checkSharedLibrariesEnabled():
      args.append('-DSEACASExodus_ENABLE_SHARED:BOOL=ON')
    return args

  def generateLibList(self, framework):
    ''' '''
    if hasattr(self.setCompilers, 'FC'):
      self.liblist = [['libexoIIv2for32.a',] + libs for libs in self.liblist] + self.liblist
      #self.liblist.append(['libexoIIv2for32.a'])
    return config.package.Package.generateLibList(self, framework)

