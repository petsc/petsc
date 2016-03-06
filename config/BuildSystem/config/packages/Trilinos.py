import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = 'origin/trilinos-release-12-6-branch'
    self.download         = ['git://https://github.com/trilinos/trilinos']
    self.downloaddirname  = 'trilinos'
    self.includes         = ['Trilinos_version.h']
    self.functions        = ['Zoltan_Create']   # one of the very few C routines in Trilinos
    self.cxx              = 1
    self.requirescxx11    = 1
    self.downloadonWindows= 0
    self.hastests         = 1
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
    self.deps            = [self.mpi,self.blasLapack]
    #
    # also requires the ./configure option --with-cxx-dialect=C++11
    return

  def formCMakeConfigureArgs(self):
    # Check for 64bit pointers
    if self.types.sizes['known-sizeof-void-p'] != 8:
      raise RuntimeError('Trilinos requires 64bit compilers!')
    # multiple libraries in Trilinos seem to depend on Boost, I cannot easily determine which
    if not self.boost.found:
      raise RuntimeError('Trilinos requires boost so add --with-boost-dir=/pathtoboost or --download-boost and run configure again')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')
    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=DEBUG')
      args.append('-DXSDK_ENABLE_DEBUG=YES')
    else:
      args.append('-DCMAKE_BUILD_TYPE=RELEASE')
      args.append('-DXSDK_ENABLE_DEBUG=NO')

    # Roscoe says I should to this
    args.append('-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION=ON')

    # Turn off single precision and complex
    args.append('-DTeuchos_ENABLE_FLOAT=OFF')
    args.append('-DTeuchos_ENABLE_COMPLEX=OFF')
    args.append('-DTpetra_INST_FLOAT=OFF')
    args.append('-DTpetra_INST_COMPLEX_FLOAT=OFF')
    args.append('-DTpetra_INST_COMPLEX_DOUBLE=OFF')

    # Trilinos cmake does not set this variable (as it should) so cmake install does not properly reset the -id and rpath of --prefix installed Trilinos libraries
    args.append('-DCMAKE_INSTALL_NAME_DIR:STRING="'+os.path.join(self.installDir,self.libdir)+'"')

    args.append('-DTPL_ENABLE_Boost=ON')
    args.append('-DTPL_Boost_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.boost.include)[2:])
    args.append('-DTPL_Boost_LIBRARIES='+self.headers.toStringNoDupes(self.boost.lib))

    args.append('-DTPL_ENABLE_MPI=ON')
    #  Below is the set of packages recommended by Mike H.
    for p in ['Epetra','AztecOO','Ifpack','Amesos2','Tpetra','Sacado','Zoltan','Stratimikos','Thyra','Isorropia','ML','Belos','Anasazi','Zoltan2','Ifpack2','ShyLU','NOX','MueLu','Stokhos','ROL','Piro','Pike','TrilinosCouplings','Panzer']:
      args.append('-DTrilinos_ENABLE_'+p+'=ON')

    args.append('-DTPL_ENABLE_GLM=OFF')

    # FEI include files cause crashes on Apple with clang compilers
    # args.append('-DTrilinos_ENABLE_fei=OFF')
    # args.append('-DTrilinos_ENABLE_Fei=OFF')
    args.append('-DTrilinos_ENABLE_FEI=OFF')

    # FEI include files cause crashes on Apple with clang compilers
    args.append('-DTrilinos_ENABLE_stk=OFF')
    args.append('-DTrilinos_ENABLE_Stk=OFF')
    args.append('-DTrilinos_ENABLE_STK=OFF')


    # The documentation specifically says:
    #     WARNING: Do not try to hack the system and set:
    #     TPL_BLAS_LIBRARIES:PATH="-L/some/dir -llib1 -llib2 ..."
    #     This is not compatible with proper CMake usage and it not guaranteed to be supported.
    # We do it anyways because the precribed way of providing the BLAS/LAPACK libraries is insane
    args.append('-DTPL_BLAS_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DTPL_LAPACK_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')

    # From the docs at http://trilinos.org/download/public-git-repository/
    #TIP: The arguments passed to CMake when building from the Trilinos public repository must include
    args.append('-DTrilinos_ASSERT_MISSING_PACKAGES=OFF')

    if self.hwloc.found:
      args.append('-DTPL_ENABLE_HWLOC:BOOL=ON')
      args.append('-DTPL_HWLOC_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.hwloc.include)[2:])
      args.append('-DTPL_HWLOC_LIBRARIES="'+self.libraries.toStringNoDupes(self.hwloc.lib)+'"')

    if self.superlu.found and self.superlu_dist.found:
      raise RuntimeError('Trilinos cannot currently support SuperLU and SuperLU_DIST in the same configuration')

    if self.superlu.found:
      args.append('-DTPL_ENABLE_SuperLU:BOOL=ON')
      args.append('-DTPL_SuperLU_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.superlu.include)[2:])
      args.append('-DTPL_SuperLU_LIBRARIES="'+self.libraries.toStringNoDupes(self.superlu.lib)+'"')

    if self.superlu_dist.found:
      args.append('-DTPL_ENABLE_SuperLUDist:BOOL=ON')
      args.append('-DTPL_SuperLUDist_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.superlu_dist.include)[2:])
      args.append('-DTPL_SuperLUDist_LIBRARIES="'+self.libraries.toStringNoDupes(self.superlu_dist.lib)+'"')

    #  Trilinos master as of commit 0eb6657d89cbe8bed1f7992956fa9b5bcfad9c44 supports only outdated versions of MUMPS
    #  with Ameso and no versions of MUMPS with Ameso2
    #if self.mumps.found:
    #  args.append('-DTPL_ENABLE_MUMPS:BOOL=ON')
    #  args.append('-DTPL_MUMPS_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.mumps.include)[2:])
    #  args.append('-DTPL_MUMPS_LIBRARIES="'+self.libraries.toStringNoDupes(self.mumps.lib+self.scalapack.lib)+'"')

    if self.mkl_pardiso.found:
      args.append('-DTPL_ENABLE_PARDISO_MKL:BOOL=ON')
      args.append('-DTPL_PARDISO_MKL_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.mkl_pardiso.include)[2:])
      args.append('-DTPL_PARDISO_MKL_LIBRARIES="'+self.libraries.toStringNoDupes(self.mkl_pardiso.lib)+'"')

    if self.parmetis.found:
      args.append('-DTPL_ENABLE_ParMETIS:BOOL=ON')
      args.append('-DTPL_ParMETIS_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.parmetis.include)[2:])
      args.append('-DTPL_ParMETIS_LIBRARIES="'+self.libraries.toStringNoDupes(self.parmetis.lib)+'"')

    if self.ptscotch.found:
      args.append('-DTPL_ENABLE_Scotch:BOOL=ON')
      args.append('-DTPL_Scotch_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.ptscotch.include)[2:])
      args.append('-DTPL_Scotch_LIBRARIES="'+self.libraries.toStringNoDupes(self.ptscotch.lib)+'"')

    if self.hypre.found:
      args.append('-DTPL_ENABLE_HYPRE:BOOL=ON')
      args.append('-DTPL_HYPRE_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.hypre.include)[2:])
      args.append('-DTPL_HYPRE_LIBRARIES="'+self.libraries.toStringNoDupes(self.hypre.lib)+'"')

    if self.hdf5.found:
      args.append('-DTPL_ENABLE_HDF5:BOOL=ON')
      args.append('-DTPL_HDF5_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.hdf5.include)[2:])
      args.append('-DTPL_HDF5_LIBRARIES="'+self.libraries.toStringNoDupes(self.hdf5.lib)+'"')
    else:
      args.append('-DTPL_ENABLE_HDF5:BOOL=OFF')

    if self.netcdf.found:
      args.append('-DTPL_ENABLE_Netcdf:BOOL=ON')
      args.append('-DTPL_Netcdf_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.netcdf.include)[2:])
      args.append('-DTPL_Netcdf_LIBRARIES="'+self.libraries.toStringNoDupes(self.netcdf.lib)+'"')
    else:
      args.append('-DTPL_ENABLE_Netcdf:BOOL=OFF')

    if self.exodusii.found:
      args.append('-DTPL_ENABLE_ExodusII:BOOL=ON')
      args.append('-DTPL_ExodusII_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.exodusii.include)[2:])
      args.append('-DTPL_ExodusII_LIBRARIES="'+self.libraries.toStringNoDupes(self.exodusii.lib)+'"')
    else:
      args.append('-DTPL_ENABLE_ExodusII:BOOL=OFF')

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
      output1,err1,ret1  = config.package.Package.executeShellCommand('make -f simplemake listlibs', timeout=25, log = self.log)
      os.unlink('simplemake')
    except RuntimeError, e:
      raise RuntimeError('Unable to generate list of Trilinos Libraries')
    # generateLibList() wants this ridiculus format
    l = output1.split(' ')
    ll = [os.path.join(dir,'lib'+l[0][2:]+'.a')]
    for i in l[1:]:
      ll.append('lib'+i[2:]+'.a')
    llp = ll
    llp.append('libpthread.a')
    return [ll,llp]


