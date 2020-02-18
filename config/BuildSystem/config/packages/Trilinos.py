import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = 'fab6197820' # xsdk-0.2.0 + superlu_dist-5.4.0, superlu_dist-6.0.1  fix oct-12-2018
    #self.download         = ['git://https://github.com/trilinos/trilinos','https://github.com/trilinos/trilinos/archive/'+self.gitcommit+'.tar.gz']
    self.download         = ['git://https://github.com/balay/trilinos','https://github.com/balay/trilinos/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames = ['trilinos']
    self.includes         = ['Trilinos_version.h']
    self.functions        = ['Zoltan_Create']   # one of the very few C routines in Trilinos
    self.cxx              = 1
    self.requirescxx11    = 1
    self.downloadonWindows= 0
    self.hastests         = 1
    self.requiresrpath    = 1
    self.precisions       = ['double']
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
    self.metis           = framework.require('config.packages.metis',self)
    self.parmetis        = framework.require('config.packages.parmetis',self)
    self.hypre           = framework.require('config.packages.hypre',self)
    self.ptscotch        = framework.require('config.packages.PTScotch',self)
    self.hdf5            = framework.require('config.packages.hdf5',self)
    self.netcdf          = framework.require('config.packages.netcdf',self)
    self.scalapack       = framework.require('config.packages.scalapack',self)
    self.mumps           = framework.require('config.packages.MUMPS',self)
    self.zoltan          = framework.require('config.packages.Zoltan',self)
    self.ml              = framework.require('config.packages.ml',self)
    self.chaco           = framework.require('config.packages.Chaco',self)
    self.exodusii        = framework.require('config.packages.exodusii',self)
    self.boost           = framework.require('config.packages.boost',self)
    self.flibs           = framework.require('config.packages.flibs',self)
    self.cxxlibs         = framework.require('config.packages.cxxlibs',self)
    self.mathlib         = framework.require('config.packages.mathlib',self)
    self.deps            = [self.mpi,self.blasLapack,self.flibs,self.cxxlibs,self.mathlib]
    self.odeps           = [self.hwloc,self.hypre,self.superlu,self.superlu_dist,self.parmetis,self.metis,self.ptscotch,self.boost,self.netcdf,self.hdf5]
    #
    # also requires the ./configure option --with-cxx-dialect=C++11
    return

  def checkTrilinosDuplicates(self):
    # we check this in two places, before doing a Trilinos install and (for when PETSc does not install Trilinos) when checking that
    # the the Trilinos libraries are valid
    if self.zoltan.found:
      raise RuntimeError('Trilinos contains Zoltan, therefore do not provide/build a Zoltan if you are providing/building Trilinos')
    if self.ml.found:
      raise RuntimeError('Trilinos contains ml, therefore do not provide/build a ml if you are providing/building Trilinos')
    if self.chaco.found:
      raise RuntimeError('Trilinos contains chaco, therefore do not provide/build a chaco if you are providing/building Trilinos')
    if self.exodusii.found:
      raise RuntimeError('Trilinos contains Exodusii, therefore do not provide/build a Exodusii if you are providing/building Trilinos')

  def configureLibrary(self):
    self.checkTrilinosDuplicates()
    config.package.Package.configureLibrary(self)
    if self.libraries.check(self.dlib, "interface"):
      self.addDefine('HAVE_CHACO',1)
      self.addDefine('HAVE_CHACO_INT_ASSIGNMENT',1)
    if self.libraries.check(self.dlib, "ML_Set_PrintLevel"):
      self.addDefine('HAVE_ML',1)
    if self.libraries.check(self.dlib, "Zoltan_LB_Partition"):
      self.addDefine('HAVE_ZOLTAN',1)
    if self.libraries.check(self.dlib, "ex_close"):
      self.addDefine('HAVE_EXODUSII',1)

  # older versions of Trilinos require passing rpath with the various library paths
  # this caused problems on Apple with cmake generating command lines that are too long
  # Trilinos was fixed to handled the rpath internally using cmake 
  def toStringNoDupes(self,string):
    string    = self.libraries.toStringNoDupes(string)
    if self.requiresrpath: return string
    newstring = ''
    for i in string.split(' '):
      if i.find('-rpath') == -1:
        newstring = newstring+' '+i
    return newstring.strip()

  def toString(self,string):
    string    = self.libraries.toString(string)
    if self.requiresrpath: return string
    newstring = ''
    for i in string.split(' '):
      if i.find('-rpath') == -1:
        newstring = newstring+' '+i
    return newstring.strip()

  def formCMakeConfigureArgs(self):
    #  Get trilinos version
    # if version is 120900 (Dev) or higher than don't require rpaths
    trequires = 0
    fd = open(os.path.join(self.packageDir,'Version.cmake'))
    bf = fd.readline()
    while bf:
      if bf.startswith('SET(Trilinos_MAJOR_MINOR_VERSION'):
        bf = bf[34:39]
        bf = int(bf)
        if bf > 120900:
          self.requiresrpath = 0
        if bf == 120900:
          trequires = 1
      if trequires:
        if bf.find('(Dev)') > -1:
          self.requirespath = 0
      bf = fd.readline()
    fd.close()

    self.checkTrilinosDuplicates()
    if not self.netcdf.found:
      raise RuntimeError('Trilinos requires netcdf so make sure you have --download-netcdf or --with-netcdf-dir if you are building Trilinos')

    if not self.hdf5.found:
      raise RuntimeError('Trilinos requires hdf5 so make sure you have --download-hdf5 or --with-hdf5-dir if you are building Trilinos')

    # Check for 64bit pointers
    if self.types.sizes['void-p'] != 8:
      raise RuntimeError('Trilinos requires 64bit compilers, your compiler is using 32 bit pointers!')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')
    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=DEBUG')
      args.append('-DTrilinos_ENABLE_DEBUG=YES')
    else:
      args.append('-DCMAKE_BUILD_TYPE=RELEASE')
      args.append('-DXSDK_ENABLE_DEBUG=NO')

    # Roscoe says I should to this
    args.append('-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION=ON')

    # Roscoe says I should set this so that any Trilinos parts that depend on missing external packages such as netcdf will be automatically turned off
    args.append('-DTrilinos_DISABLE_ENABLED_FORWARD_DEP_PACKAGES=ON')

    args.append('-DTrilinos_EXTRA_LINK_FLAGS="'+self.toStringNoDupes(self.flibs.lib+self.cxxlibs.lib+self.mathlib.lib)+' '+self.compilers.LIBS+'"')

    # Turn off single precision and complex
    args.append('-DTeuchos_ENABLE_FLOAT=OFF')
    args.append('-DTeuchos_ENABLE_COMPLEX=OFF')
    args.append('-DTpetra_INST_FLOAT=OFF')
    args.append('-DTpetra_INST_COMPLEX_FLOAT=OFF')
    args.append('-DTpetra_INST_COMPLEX_DOUBLE=OFF')

    # Trilinos cmake does not set this variable (as it should) so cmake install does not properly reset the -id and rpath of --prefix installed Trilinos libraries
    args.append('-DCMAKE_INSTALL_NAME_DIR:STRING="'+os.path.join(self.installDir,self.libdir)+'"')

    if self.boost.found:
      args.append('-DTPL_ENABLE_Boost=ON')
      args.append('-DTPL_Boost_INCLUDE_DIRS="'+';'.join(self.boost.include)+'"')
      args.append('-DTPL_Boost_LIBRARIES='+self.headers.toStringNoDupes(self.boost.lib))
    else:
      args.append('-DTPL_ENABLE_TPL_Boost:BOOL=OFF')
      args.append('-DTPL_ENABLE_TPL_BoostLib:BOOL=OFF')

    args.append('-DTPL_ENABLE_MPI=ON')
    #  Below is the set of packages recommended by Mike H.
    for p in ['Epetra','AztecOO','Ifpack','Amesos2','Tpetra','Sacado','Zoltan','Stratimikos','Thyra','Isorropia','ML','Belos','Anasazi','Zoltan2','Ifpack2','ShyLU','NOX','MueLu','Stokhos','ROL','Piro','Pike','TrilinosCouplings','Panzer','SEACAS']:
      args.append('-DTrilinos_ENABLE_'+p+'=ON')

    # SEACAS which contains Exodusii needs to have the following turned off
    args.append('-DTPL_ENABLE_Matio=OFF')
    args.append('-DTPL_ENABLE_GLM=OFF')

    # SEACAS finds an X11 to use but this can fail on some machines like the Cray
    args.append('-DTPL_ENABLE_X11=OFF')

    # FEI include files cause crashes on Apple with clang compilers
    # args.append('-DTrilinos_ENABLE_fei=OFF')
    # args.append('-DTrilinos_ENABLE_Fei=OFF')
    args.append('-DTrilinos_ENABLE_FEI=OFF')

    # FEI include files cause crashes on Apple with clang compilers
    args.append('-DTrilinos_ENABLE_STK=OFF')

    if not hasattr(self.compilers, 'FC'):
      args.append('-DTrilinos_ENABLE_Fortran=OFF')

    # The documentation specifically says:
    #     WARNING: Do not try to hack the system and set:
    #     TPL_BLAS_LIBRARIES:PATH="-L/some/dir -llib1 -llib2 ..."
    #     This is not compatible with proper CMake usage and it not guaranteed to be supported.
    # We do it anyways because the precribed way of providing the BLAS/LAPACK libraries is insane
    args.append('-DTPL_BLAS_LIBRARIES="'+self.toString(self.blasLapack.dlib)+'"')
    args.append('-DTPL_LAPACK_LIBRARIES="'+self.toString(self.blasLapack.dlib)+'"')

    # From the docs at http://trilinos.org/download/public-git-repository/
    #TIP: The arguments passed to CMake when building from the Trilinos public repository must include
    args.append('-DTrilinos_ASSERT_MISSING_PACKAGES=OFF')

    if self.hwloc.found:
      args.append('-DTPL_ENABLE_HWLOC:BOOL=ON')
      args.append('-DTPL_HWLOC_INCLUDE_DIRS="'+';'.join(self.hwloc.include)+'"')
      args.append('-DTPL_HWLOC_LIBRARIES="'+self.toStringNoDupes(self.hwloc.lib)+'"')

    if self.superlu.found and self.superlu_dist.found:
      raise RuntimeError('Trilinos cannot currently support SuperLU and SuperLU_DIST in the same configuration')

    if self.superlu.found:
      args.append('-DTPL_ENABLE_SuperLU:BOOL=ON')
      args.append('-DTPL_SuperLU_INCLUDE_DIRS="'+';'.join(self.superlu.include)+'"')
      args.append('-DTPL_SuperLU_LIBRARIES="'+self.toStringNoDupes(self.superlu.lib)+'"')
    else:
      args.append('-DTPL_ENABLE_TPL_SuperLU:BOOL=OFF')

    if self.superlu_dist.found:
      args.append('-DTPL_ENABLE_SuperLUDist:BOOL=ON')
      args.append('-DTPL_SuperLUDist_INCLUDE_DIRS="'+';'.join(self.superlu_dist.include)+'"')
      args.append('-DTPL_SuperLUDist_LIBRARIES="'+self.toStringNoDupes(self.superlu_dist.lib)+'"')
    else:
      args.append('-DTPL_ENABLE_TPL_SuperLUDist:BOOL=OFF')

    if self.hypre.found:
      args.append('-DTPL_ENABLE_HYPRE:BOOL=ON')
      args.append('-DTPL_HYPRE_INCLUDE_DIRS="'+';'.join(self.hypre.include)+'"')
      args.append('-DTPL_HYPRE_LIBRARIES="'+self.toStringNoDupes(self.hypre.lib)+'"')
    else:
      args.append('-DTPL_ENABLE_HYPRE:BOOL=OFF')

    #  Trilinos master as of commit 0eb6657d89cbe8bed1f7992956fa9b5bcfad9c44 supports only outdated versions of MUMPS
    #  with Ameso and no versions of MUMPS with Ameso2
    #if self.mumps.found:
    #  args.append('-DTPL_ENABLE_MUMPS:BOOL=ON')
    #  args.append('-DTPL_MUMPS_INCLUDE_DIRS="'+';'.join(self.mumps.include)+'"')
    #  args.append('-DTPL_MUMPS_LIBRARIES="'+self.toStringNoDupes(self.mumps.lib+self.scalapack.lib)+'"')

    if self.mkl_pardiso.found:
      args.append('-DTPL_ENABLE_PARDISO_MKL:BOOL=ON')
      args.append('-DTPL_PARDISO_MKL_INCLUDE_DIRS="'+';'.join(self.mkl_pardiso.include)+'"')
      args.append('-DTPL_PARDISO_MKL_LIBRARIES="'+self.toStringNoDupes(self.mkl_pardiso.lib)+'"')
    else:
      args.append('-DTPL_ENABLE_TPL_PARDISO_MKL:BOOL=OFF')

    if self.metis.found:
      args.append('-DTPL_ENABLE_METIS:BOOL=ON')
      args.append('-DTPL_METIS_INCLUDE_DIRS="'+';'.join(self.metis.include)+'"')
      args.append('-DTPL_METIS_LIBRARIES="'+self.toStringNoDupes(self.metis.lib)+'"')
    else:
      args.append('-DTPL_ENABLE_METIS:BOOL=OFF')

    if self.parmetis.found:
      args.append('-DTPL_ENABLE_ParMETIS:BOOL=ON')
      args.append('-DTPL_ParMETIS_INCLUDE_DIRS="'+';'.join(self.parmetis.include)+'"')
      args.append('-DTPL_ParMETIS_LIBRARIES="'+self.toStringNoDupes(self.parmetis.lib)+'"')
    else:
      args.append('-DTPL_ENABLE_ParMETIS:BOOL=OFF')

    if self.ptscotch.found:
      args.append('-DTPL_ENABLE_Scotch:BOOL=ON')
      args.append('-DTPL_Scotch_INCLUDE_DIRS="'+';'.join(self.ptscotch.include)+'"')
      args.append('-DTPL_Scotch_LIBRARIES="'+self.toStringNoDupes(self.ptscotch.lib)+'"')
    else:
      args.append('-DTPL_ENABLE_Scotch:BOOL=OFF')

    if self.hdf5.found:
      args.append('-DTPL_ENABLE_HDF5:BOOL=ON')
      args.append('-DTPL_HDF5_INCLUDE_DIRS="'+';'.join(self.hdf5.include)+'"')
      args.append('-DTPL_HDF5_LIBRARIES="'+self.toStringNoDupes(self.hdf5.dlib)+'"')
    else:
      args.append('-DTPL_ENABLE_HDF5:BOOL=OFF')

    if self.netcdf.found:
      args.append('-DTPL_ENABLE_Netcdf:BOOL=ON')
      args.append('-DTPL_Netcdf_INCLUDE_DIRS="'+';'.join(self.netcdf.include)+'"')
      args.append('-DTPL_Netcdf_LIBRARIES="'+self.toStringNoDupes(self.netcdf.dlib)+'"')
    else:
      args.append('-DTPL_ENABLE_Netcdf:BOOL=OFF')

    args.append('-DTPL_ENABLE_ExodusII:BOOL=OFF')

    if not hasattr(self.setCompilers, 'FC'):
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

