import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version         = '3.0.0'
    self.minversion      = '2.14'
    self.versionname     = 'HYPRE_RELEASE_VERSION'
    self.versioninclude  = 'HYPRE_config.h'
    self.requiresversion = 1
    self.gitcommit       = 'v'+self.version
    self.download        = ['git://https://github.com/hypre-space/hypre','https://github.com/hypre-space/hypre/archive/'+self.gitcommit+'.tar.gz']
    self.functions       = ['HYPRE_IJMatrixCreate']
    self.includes        = ['HYPRE.h']
    self.liblist         = [['libHYPRE.a']]
    self.buildLanguages  = ['C','Cxx']
    self.precisions        = ['single', 'double', '__float128']
    self.hastests          = 1
    self.hastestsdatafiles = 1

  def setupHelp(self, help):
    config.package.GNUPackage.setupHelp(self,help)
    import nargs
    help.addArgument('HYPRE', '-with-hypre-gpu-arch=<string>',  nargs.ArgString(None, 0, 'Value passed to hypre\'s --with-gpu-arch= configure option'))
    help.addArgument('HYPRE', '-download-hypre-openmp', nargs.ArgBool(None, 1, 'Let hypre use OpenMP if available'))
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.openmp        = framework.require('config.packages.OpenMP',self)
    self.cxxlibs       = framework.require('config.packages.cxxlibs',self)
    self.blasLapack    = framework.require('config.packages.BlasLapack',self)
    self.mpi           = framework.require('config.packages.MPI',self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.cuda          = framework.require('config.packages.CUDA',self)
    self.hip           = framework.require('config.packages.HIP',self)
    self.sycl          = framework.require('config.packages.SYCL',self)
    self.umpire        = framework.require('config.packages.Umpire',self)
    self.openmp        = framework.require('config.packages.OpenMP',self)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.scalar        = framework.require('PETSc.options.scalarTypes',self)
    self.languages     = framework.require('PETSc.options.languages',self)
    self.deps          = [self.mpi,self.blasLapack,self.cxxlibs,self.mathlib]
    self.odeps         = [self.cuda,self.hip,self.openmp,self.umpire]
    if self.setCompilers.isCrayKNL(None,self.log):
      self.installwithbatch = 0

  def formGNUConfigureArgs(self):
    self.packageDir = os.path.join(self.packageDir,'src')
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if not hasattr(self.compilers, 'FC'):
      args.append('--disable-fortran')
    if self.mpi.usingMPIUni:
      args.append('--without-MPI')
    elif self.mpi.include:
      # just use the first dir - and assume the subsequent one isn't necessary [relevant only on AIX?]
      args.append('--with-MPI-include="'+self.mpi.include[0]+'"')
    libdirs = []
    for l in self.mpi.lib:
      ll = os.path.dirname(l)
      libdirs.append(ll)
    libdirs = ' '.join(libdirs)
    args.append('--with-MPI-lib-dirs="'+libdirs+'"')
    libs = []
    for l in self.mpi.lib:
      ll = os.path.basename(l)
      if ll.endswith('.a'): libs.append(ll[3:-2])
      if ll.endswith('.so'): libs.append(ll[3:-3])
      if ll.endswith('.dylib'): libs.append(ll[3:-6])
    libs = ' '.join(libs)
    args.append('--with-MPI-libs="'+libs+'"')

    # tell hypre configure not to look for blas/lapack [and not use hypre-internal blas]
    args.append('--with-blas-lib="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('--with-lapack-lib=" "')
    args.append('--with-blas=no')
    args.append('--with-lapack=no')

    # floating point precisions
    if self.scalar.precision == 'single':
      args.append('--enable-single')
    elif self.scalar.precision == '__float128':
      args.append('--enable-longdouble')

    # HYPRE automatically detects essl symbols and includes essl.h!
    # There are no configure options to disable it programmatically
    if hasattr(self.blasLapack,'essl'):
      args = self.addArgStartsWith(args,'CFLAGS',self.headers.toString(self.blasLapack.include))
      args = self.addArgStartsWith(args,'CXXFLAGS',self.headers.toString(self.blasLapack.include))

    # device configuration
    cucc = ''
    devflags = ''
    hipbuild = False
    cudabuild = False
    syclbuild = False
    hasharch = 'with-gpu-arch' in args
    if self.hip.found:
      stdflag  = '-std=c++14'
      hipbuild = True
      args.append('ROCM_PATH="{0}"'.format(self.hip.hipDir))
      args.append('--with-hip')
      if not hasharch:
        if not 'with-hypre-gpu-arch' in self.framework.clArgDB:
          if hasattr(self.hip,'hipArch'):
            args.append('--with-gpu-arch=' + self.hip.hipArch)
          else:
            args.append('--with-gpu-arch=gfx908') # defaults to MI100
        else:
          args.append('--with-gpu-arch='+self.argDB['with-hypre-gpu-arch'])
      self.pushLanguage('HIP')
      cucc = self.getCompiler()
      devflags += ' '.join(('','-x','hip',stdflag,''))
      devflags += ' '.join(self.removeVisibilityFlag(self.getCompilerFlags().split())) + ' ' + self.setCompilers.HIPPPFLAGS + ' ' + self.mpi.includepaths + ' ' + self.headers.toString(self.dinclude)
      self.popLanguage()
    elif self.cuda.found:
      stdflag   = '-std=c++17'
      cudabuild = True
      if not hasattr(self.cuda, 'cudaDir'):
        raise RuntimeError('CUDA directory not detected! Mail configure.log to petsc-maint@mcs.anl.gov.')
      args.append('CUDA_HOME="'+self.cuda.cudaDir+'"')
      args.append('--with-cuda')
      if not hasharch:
        if not 'with-hypre-gpu-arch' in self.framework.clArgDB:
          if hasattr(self.cuda,'cudaArch'):
            args.append('--with-gpu-arch="' + ' '.join(self.cuda.cudaArchList()) + '"')
          else:
            args.append('--with-gpu-arch=70') # default
        else:
          args.append('--with-gpu-arch='+self.argDB['with-hypre-gpu-arch'])
      self.pushLanguage('CUDA')
      cucc = self.getCompiler()
      devflags += ' '.join(('','-expt-extended-lambda',stdflag,'-x','cu',''))
      devflags += self.updatePackageCUDAFlags(self.getCompilerFlags()) + ' ' + self.setCompilers.CUDAPPFLAGS + ' ' + self.mpi.includepaths+ ' ' + self.headers.toString(self.dinclude)
      self.popLanguage()
    elif self.sycl.found:
      syclbuild = True
      args.append('--with-sycl')
      if hasattr(self.sycl, 'targets'):
        args.append('--with-sycl-target='+self.sycl.targets)
      if hasattr(self.sycl, 'syclArch'):
        # e.g., --with-sycl-target-backend="'-device pvc'"
        args.append('--with-sycl-target-backend="\'-device {dev}\'"'.format(dev=self.sycl.syclArch))
      self.pushLanguage(self.languages.clanguage) # If with sycl, PETSc's build compiler must be a sycl compiler
      cucc = self.getCompiler()
      devflags += ' '.join(self.removeVisibilityFlag(self.getCompilerFlags().split())) + ' ' + self.setCompilers.SYCLPPFLAGS + ' ' + self.mpi.includepaths + ' ' + self.headers.toString(self.dinclude)
      self.popLanguage()
    elif self.openmp.found and self.argDB['download-hypre-openmp']:
      args.append('--with-openmp')
      self.usesopenmp = 'yes'

    # If building for CUDA or HIP, checks whether Umpire is available or not
    if (cudabuild or hipbuild):
      if not hasattr(self, 'umpire') or not self.umpire.found:
        self.logPrintWarning('Compiling HYPRE with GPU support but without Umpire support (not recommended). For best performance, consider reconfiguring --with-umpire-dir or with --download-umpire')
        args.append('--without-umpire')
      else:
        args.append('--with-umpire')
        args.append('--with-umpire-include="'+self.umpire.include[0]+'"')
        args.append('--with-umpire-lib="'+self.libraries.toString(self.umpire.dlib)+'"')
        args.append('--with-umpire-lib-dirs=')

    if self.usesopenmp == 'no':
      if hasattr(self,'openmp') and hasattr(self.openmp,'ompflag'):
        args = self.rmValueArgStartsWith(args,['CC','CXX','FC'],self.openmp.ompflag)

    if cucc:
      args.append('CUCC="'+cucc+'"')
      args.append('CUFLAGS="'+devflags+'"')

    # explicitly tell hypre BLAS/LAPACK mangling since it may not match Fortran mangling
    if self.blasLapack.mangling == 'underscore':
      mang = 'one-underscore'
    elif self.blasLapack.mangling == 'caps':
      mang = 'caps-no-underscores'
    else:
      mang = 'no-underscores'
    args.append('--with-fmangle-blas='+mang)
    args.append('--with-fmangle-lapack='+mang)

    args.append('--without-mli')
    args.append('--without-superlu')

    if self.getDefaultIndexSize() == 64:
      if cudabuild or hipbuild or syclbuild: # HYPRE 2.23 supports only mixedint configurations with CUDA/HIP/SYCL
        args.append('--enable-bigint=no --enable-mixedint=yes')
      else:
        args.append('--enable-bigint')
    if self.scalar.scalartype == 'complex':
      args.append('--enable-complex')

    # hypre configure assumes the AR flags are passed in with AR
    args = self.rmArgsStartsWith(args,['AR'])
    args.append('AR="'+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'"')

    # On CRAY with shared libraries, libHYPRE.so is linked as
    # $ cc -shared -o libHYPRE.so ...a bunch of .o files.... ...libraries.... -dynamic
    # The -dynamic at the end makes cc think it is creating an executable
    args = self.rmArgsStartsWith(args,['LDFLAGS'])
    args.append('LDFLAGS="'+self.setCompilers.LDFLAGS.replace('-dynamic','')+'"')

    # Prevent NVCC from complaining about different standards
    if cudabuild or hipbuild:
      for dialect in ('20','17','14','11'):
        if dialect < stdflag[-2:]:
          break
        gnuflag = '-std=gnu++'+dialect
        cppflag = '-std=c++'+dialect
        args    = [a.replace(gnuflag,stdflag).replace(cppflag,stdflag) for a in args]

    # On MSYS2, need to bypass the following error by forcing the architecture
    # configure:2898: checking host system type
    # configure:2911: result:
    # configure:2915: error: invalid value of canonical host
    if 'MSYSTEM' in os.environ and os.environ['MSYSTEM'].endswith('64'):
      args.append('--host=x86_64-linux-gnu')

    self.logPrintBox('hypre examples are available at '+os.path.join(self.packageDir,'examples'))
    return args

  def consistencyChecks(self):
    config.package.GNUPackage.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      if not self.blasLapack.checkForRoutine('dgels'):
        raise RuntimeError('hypre requires the LAPACK routine dgels(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it')
      self.log.write('Found dgels() in Lapack library as needed by hypre\n')
    return

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    flagsArg = self.getPreprocessorFlagsArg()
    oldFlags = getattr(self.compilers, flagsArg)
    setattr(self.compilers, flagsArg, oldFlags+' '+self.headers.toString(self.include))
    # check complex/real
    scn  = '!' if self.scalar.scalartype == 'complex' else ''
    code = '#if {0}defined(HYPRE_COMPLEX)\n#error Mismatch between HYPRE and PETSc scalar types\n#endif'.format(scn)
    if not self.checkCompile('#include "HYPRE_config.h"',code):
      msg  = 'HYPRE scalar numbers configuration is different than the requested type {0}\n'.format(self.scalar.scalartype)
      raise RuntimeError('Hypre specified is incompatible!\n'+msg+'Suggest using --download-hypre for a compatible hypre')
    # check precision
    b = self.scalar.precisionToBytes()
    size = self.types.checkSizeof('HYPRE_Real', (8, 4, 16), otherInclude='HYPRE_utilities.h', save=False)
    if size != b:
      msg  = 'HYPRE Real numbers configuration is incompatible with the requested precision {0}\n'.format(self.scalar.precision)
      raise RuntimeError('Hypre specified is incompatible!\n'+msg+'Suggest using --download-hypre for a compatible hypre')
    # check integers
    if self.defaultIndexSize == 64:
      code = '#if !defined(HYPRE_BIGINT) && !defined(HYPRE_MIXEDINT)\n#error HYPRE_BIGINT or HYPRE_MIXEDINT not defined!\n#endif'
      msg  = '--with-64-bit-indices option requires Hypre built with --enable-bigint or --enable-mixedint.\n'
    else:
      code = '#if defined(HYPRE_BIGINT)\n#error HYPRE_BIGINT defined!\n#endif\n#if defined(HYPRE_MIXEDINT)\n#error HYPRE_MIXEDINT defined!\n#endif\n'
      msg  = 'Hypre with --enable-bigint/--enable-mixedint appears to be specified for a 32-bit-indices build of PETSc.\n'
    if not self.checkCompile('#include "HYPRE_config.h"',code):
      raise RuntimeError('Hypre specified is incompatible!\n'+msg+'Suggest using --download-hypre for a compatible hypre')
    code = '#if defined(HYPRE_MIXEDINT)\n#error HYPRE_MIXEDINT defined!\n#endif\n'
    if not self.checkCompile('#include "HYPRE_config.h"',code):
      self.addDefine('HAVE_HYPRE_MIXEDINT', 1)
    code = '#if defined(HYPRE_USING_GPU)\n#error HYPRE_USING_GPU defined!\n#endif\n'
    if not self.checkCompile('#include "HYPRE_config.h"',code):
      self.addDefine('HAVE_HYPRE_DEVICE', 1)
    setattr(self.compilers, flagsArg,oldFlags)
    return
