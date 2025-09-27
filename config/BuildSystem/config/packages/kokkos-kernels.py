import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '4.7.01'
    self.minversion       = '3.7.01'
    self.versionname      = 'KOKKOSKERNELS_VERSION'
    self.download         = ['git://https://github.com/kokkos/kokkos-kernels.git','https://github.com/kokkos/kokkos-kernels/archive/'+self.gitcommit+'.tar.gz']
    self.includes         = ['KokkosBlas.hpp','KokkosSparse_CrsMatrix.hpp']
    self.liblist          = [['libkokkoskernels.a']]
    self.functions        = ['']
    # Even libkokkoskernels exists, we really don't know which KK components are enabled and which functions/symbols are there
    self.functionsCxx     = [1,'#include <iostream>','std::cout << "Assume Kokkos-Kernels is header only and skip the function test";']
    self.buildLanguages   = ['Cxx']
    self.hastests         = 1
    self.requiresrpath    = 1
    self.minCmakeVersion  = (3,10,0)
    return

  def __str__(self):
    output  = config.package.CMakePackage.__str__(self)
    if hasattr(self,'system'): output += '  Backend: '+self.system+'\n'
    return output

  def setupHelp(self, help):
    import nargs
    config.package.CMakePackage.setupHelp(self, help)
    help.addArgument('KOKKOS-KERNELS', '-with-kokkos-kernels-tpl=<bool>', nargs.ArgBool(None, 1, 'Indicate if you wish to let Kokkos-Kernels use Third-Party Libraries (TPLs)'))
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.externalpackagesdir = framework.require('PETSc.options.externalpackagesdir',self)
    self.scalarTypes         = framework.require('PETSc.options.scalarTypes',self)
    self.kokkos              = framework.require('config.packages.kokkos',self)
    self.deps                = [self.kokkos]
    self.cuda                = framework.require('config.packages.CUDA',self)
    self.hip                 = framework.require('config.packages.HIP',self)
    self.sycl                = framework.require('config.packages.SYCL',self)
    self.blasLapack          = framework.require('config.packages.BlasLapack',self)
    self.odeps               = [self.cuda,self.hip,self.sycl,self.blasLapack]
    return

  def versionToStandardForm(self,ver):
    '''Converts from Kokkos kernels 30101 notation to standard notation 3.1.01'''
    return ".".join(map(str,[int(ver)//10000, int(ver)//100%100, int(ver)%100]))

  def toString(self,string):
    string    = self.libraries.toString(string)
    if self.requiresrpath: return string
    newstring = ''
    for i in string.split(' '):
      if i.find('-rpath') == -1:
        newstring = newstring+' '+i
    return newstring.strip()

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    KokkosRoot = self.kokkos.directory
    args.append('-DKokkos_ROOT='+KokkosRoot)
    if self.scalarTypes.scalartype == 'complex':
      if self.scalarTypes.precision == 'double':
        args.append('-DKokkosKernels_INST_COMPLEX_DOUBLE=ON')
      elif self.scalarTypes.precision == 'single':
        args.append('-DKokkosKernels_INST_COMPLEX_FLOAT=ON')

    if self.cuda.found:
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
      if self.cuda.cudaclang:
        args.append('-DCMAKE_CXX_COMPILER='+self.getCompiler('CUDA'))
      else:
        args.append('-DCMAKE_CXX_COMPILER='+self.getCompiler('Cxx')) # use the host CXX compiler, let Kokkos handle the nvcc_wrapper business
      if not self.argDB['with-kokkos-kernels-tpl']:
        args.append('-DKokkosKernels_ENABLE_TPL_CUBLAS=OFF')  # These are turned ON by KK by default when CUDA is enabled
        args.append('-DKokkosKernels_ENABLE_TPL_CUSPARSE=OFF')
        args.append('-DKokkosKernels_ENABLE_TPL_CUSOLVER=OFF')
      elif hasattr(self.cuda, 'math_libs_dir'): # KK-4.3+ failed to locate nvhpc math_libs on Perlmutter@NERSC, so we set them explicitly
        args.append('-DCUBLAS_ROOT='+self.cuda.math_libs_dir)
        args.append('-DCUSPARSE_ROOT='+self.cuda.math_libs_dir)
        args.append('-DCUSOLVER_ROOT='+self.cuda.math_libs_dir)
    elif self.hip.found:
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
      args.append('-DCMAKE_CXX_COMPILER='+self.getCompiler('HIP'))
      # TPL
      if self.argDB['with-kokkos-kernels-tpl'] and os.path.isdir(self.hip.rocBlasDir) and os.path.isdir(self.hip.rocSparseDir): # TPL is required either by default or by users
        args.append('-DKokkosKernels_ENABLE_TPL_ROCBLAS=ON')
        args.append('-DKokkosKernels_ENABLE_TPL_ROCSPARSE=ON')
        args.append('-DKokkosKernels_ENABLE_TPL_ROCSOLVER=ON')
        # rocm re-organized directory since 6.0.0
        #   Before 6.0.0: /opt/rocm-5.4.3/include/{rocblas.h, rocsparse.h, ...}, /opt/rocm-5.4.3/lib/{librocblas.so, librocsparse.so, ...}
        #   Since 6.0.0: /opt/rocm-6.0.0/include/{rocblas/rocblas.h, rocsparse/rocsparse.h, ...}, /opt/rocm-6.0.0/lib/{librocblas.so, librocsparse.so, ...}
        # KK-4.5.1 failed with the simple -DROCBLAS_ROOT=/opt/rocm-6.0.0, so we go verbosely
        if self.hip.version_tuple >= (6, 0, 0):
          args.append('-DROCBLAS_LIBRARIES=rocblas')
          args.append('-DROCBLAS_LIBRARY_DIRS='+os.path.join(self.hip.hipDir, 'lib'))
          args.append('-DROCBLAS_INCLUDE_DIRS='+os.path.join(self.hip.hipDir, 'include', 'rocblas'))
          args.append('-DROCSPARSE_LIBRARIES=rocsparse')
          args.append('-DROCSPARSE_LIBRARY_DIRS='+os.path.join(self.hip.hipDir, 'lib'))
          args.append('-DROCSPARSE_INCLUDE_DIRS='+os.path.join(self.hip.hipDir, 'include', 'rocsparse'))
          args.append('-DROCSOLVER_LIBRARIES=rocsolver')
          args.append('-DROCSOLVER_LIBRARY_DIRS='+os.path.join(self.hip.hipDir, 'lib'))
          args.append('-DROCSOLVER_INCLUDE_DIRS='+os.path.join(self.hip.hipDir, 'include', 'rocsolver'))
        else:
          args.append('-DROCBLAS_ROOT='+self.hip.hipDir) # KK-4.0.1 and higher only support these
          args.append('-DROCSPARSE_ROOT='+self.hip.hipDir)
          args.append('-DROCSPARSE_ROOT='+self.hip.hipDir)
          args.append('-DKokkosKernels_ROCBLAS_ROOT='+self.hip.hipDir) # KK-4.0.0 and lower support these; remove the two lines once self.miniversion >= 4.0.1
          args.append('-DKokkosKernels_ROCSPARSE_ROOT='+self.hip.hipDir)
          args.append('-DKokkosKernels_ROCSOLVER_ROOT='+self.hip.hipDir)
      elif 'with-kokkos-kernels-tpl' in self.framework.clArgDB and self.argDB['with-kokkos-kernels-tpl']: # TPL is explicitly required by users
        raise RuntimeError('Kokkos-Kernels TPL is required but {x} and {y} do not exist! If not needed, use --with-kokkos-kernels-tpl=0'.format(x=self.hip.rocBlasDir,y=self.hip.rocSparseDir))
      else: # Users turned it off or because rocBlas/rocSparse dirs not found
        args.append('-DKokkosKernels_ENABLE_TPL_ROCBLAS=OFF')
        args.append('-DKokkosKernels_ENABLE_TPL_ROCSPARSE=OFF')
        args.append('-DKokkosKernels_ENABLE_TPL_ROCSOLVER=OFF')
    elif self.sycl.found:
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
      args.append('-DCMAKE_CXX_COMPILER='+self.getCompiler('SYCL'))
      if self.argDB['with-kokkos-kernels-tpl']:
        if self.blasLapack.mkl: # KK uses them to find MKL
          args.append('-DKokkosKernels_ENABLE_TPL_MKL=ON')
        elif 'with-kokkos-kernels-tpl' in self.framework.clArgDB:
          raise RuntimeError('Kokkos-Kernels TPL is explicitly required but could not find OneMKL')

    # These options will be taken from Kokkos configuration
    args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_STANDARD=')
    args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_FLAGS')
    args = self.rmArgsStartsWith(args,'-DCMAKE_C_COMPILER=')
    args = self.rmArgsStartsWith(args,'-DCMAKE_C_FLAGS')
    args = self.rmArgsStartsWith(args,'-DCMAKE_AR')
    args = self.rmArgsStartsWith(args,'-DCMAKE_RANLIB')
    return args

  def configureLibrary(self):
    needRestore = False
    self.buildLanguages= self.kokkos.buildLanguages
    if self.cuda.found and not self.cuda.cudaclang:
        oldFlags = self.setCompilers.CUDAPPFLAGS
        self.setCompilers.CUDAPPFLAGS += " -ccbin " + self.getCompiler('Cxx')
        needRestore = True

    config.package.CMakePackage.configureLibrary(self)

    if needRestore: self.setCompilers.CUDAPPFLAGS = oldFlags
