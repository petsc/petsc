import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = 'origin/master'
    self.download         = ['hg://https://software.lanl.gov/ascem/hg/amanzi-ideas']
    self.downloaddirnames = ['amanzi-ideas']
    self.includes         = []
    self.functions        = []
    self.cxx              = 1
    self.requirescxx11    = 1
    self.downloadonWindows= 0
    self.hastests         = 1
    self.useddirectly     = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.superlu         = framework.require('config.packages.SuperLU',self)
    self.superlu_dist    = framework.require('config.packages.SuperLU_DIST',self)
    self.mkl_pardiso     = framework.require('config.packages.mkl_pardiso',self)
    self.parmetis        = framework.require('config.packages.parmetis',self)
    self.ptscotch        = framework.require('config.packages.PTScotch',self)
    self.hypre           = framework.require('config.packages.hypre',self)
    self.hdf5            = framework.require('config.packages.hdf5',self)
    self.netcdf          = framework.require('config.packages.netcdf',self)
    self.exodusii        = framework.require('config.packages.exodusii',self)
    self.scalapack       = framework.require('config.packages.scalapack',self)
    self.mumps           = framework.require('config.packages.MUMPS',self)
    self.boost           = framework.require('config.packages.boost',self)

    self.pflotran        = framework.require('config.packages.pflotran',self)
    self.alquimia        = framework.require('config.packages.alquimia',self)
    self.mstk            = framework.require('config.packages.mstk',self)
    self.ascem_io        = framework.require('config.packages.ascem-io',self)
    self.trilinos        = framework.require('config.packages.Trilinos',self)
    self.unittestcpp     = framework.require('config.packages.unittestcpp',self)
    self.deps            = [self.mpi,self.blasLapack]
    #
    # also requires the ./configure option --with-cxx-dialect=C++11
    return

  def formCMakeConfigureArgs(self):

#The use if include[0] etc to find the prefix path for the various packages is terrible, each xxx.py package should
#properly set the prefix path as a variable that could be used

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')
    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=DEBUG')
      args.append('-DXSDK_ENABLE_DEBUG=YES')
    else:
      args.append('-DCMAKE_BUILD_TYPE=RELEASE')
      args.append('-DXSDK_ENABLE_DEBUG=NO')

    # This is required for CMAKE to properly make shared libraries on Apple
    args.append('-DCMAKE_INSTALL_NAME_DIR:STRING="'+os.path.join(self.installDir,self.libdir)+'"')

    if self.boost.found:
      args.append('-DBOOST_INCLUDEDIR='+os.path.join(self.boost.directory,self.boost.includedir))
      args.append('-DBOOST_LIBRARYDIR='+os.path.join(self.boost.directory,self.boost.libdir))
    else:
      raise RuntimeError("Amanzi requires Boost")

    if self.hdf5.found:
      if self.hdf5.include:
        args.append('-DHDF5_ROOT='+os.path.dirname(self.hdf5.include[0]))
    else:
      raise RuntimeError("Amanzi requires HDF5")

    args.append('-DTrilinos_DIR:FILEPATH='+os.path.dirname(self.trilinos.include[0]))

    args.append('-DUnitTest_LIBRARIES="'+self.libraries.toStringNoDupes(self.unittestcpp.lib)+'"')
    args.append('-DUnitTest_INCLUDE_DIRS='+os.path.join(self.unittestcpp.directory,'include','UnitTest++','UnitTest++'))

    self.framework.pushLanguage('C')
    args.append('-DMPI_C_COMPILER="'+self.framework.getCompiler()+'"')
    self.framework.popLanguage()

    self.framework.pushLanguage('Cxx')
    args.append('-DMPI_CXX_COMPILER="'+self.framework.getCompiler()+'"')
    self.framework.popLanguage()

    if hasattr(self.setCompilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('-DMPI_Fortran_COMPILER="'+self.framework.getCompiler()+'"')
      self.framework.popLanguage()
    else:
      args.append('-DTrilinos_ENABLE_Fortran=OFF')

    return args

  def generateLibList(self,dir):
    import os
    '''Gets the complete list of Trilinos libraries'''
    fd = open('simplemake','w')
    fd.write('include '+os.path.join(dir,'..','include','Makefile.export.Trilinos')+'\n')
    fd.write('listlibs:\n\t-@echo ${Trilinos_LIBRARIES}')
    fd.close()
    try:
      output1,err1,ret1  = config.package.Package.executeShellCommand('make -f simplemake listlibs', timeout=60, log = self.log)
      os.unlink('simplemake')
    except RuntimeError as e:
      raise RuntimeError('Unable to generate list of Trilinos Libraries')
    # generateLibList() wants this ridiculus format
    l = output1.split(' ')
    ll = [os.path.join(dir,'lib'+l[0][2:]+'.a')]
    for i in l[1:]:
      ll.append('lib'+i[2:]+'.a')
    llp = ll
    llp.append('libpthread.a')
    return [ll,llp]


