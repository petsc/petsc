import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version         = '2.22.1'
    self.minversion      = '2.14'
    self.versionname     = 'HYPRE_RELEASE_VERSION'
    self.versioninclude  = 'HYPRE_config.h'
    self.requiresversion = 1
    # self.gitcommit       = 'v'+self.version
    self.gitcommit       = 'eaff5505ed08e66a9c138cae6b0bc0a390232f83' # master Sep 9, 2021 GPU support with PETSc bugfixes
    self.download        = ['git://https://github.com/hypre-space/hypre','https://github.com/hypre-space/hypre/archive/'+self.gitcommit+'.tar.gz']
    self.functions       = ['HYPRE_IJMatrixCreate']
    self.includes        = ['HYPRE.h']
    self.liblist         = [['libHYPRE.a']]
    self.license         = 'https://computation.llnl.gov/casc/linear_solvers/sls_hypre.html'
    # Per hypre users guide section 7.5 - install manually on windows for MS compilers.
    self.downloadonWindows = 0
    self.precisions        = ['double']
    # HYPRE is supposed to work with complex number
    #self.complex           = 0
    self.hastests          = 1
    self.hastestsdatafiles = 1

  def setupHelp(self, help):
    config.package.GNUPackage.setupHelp(self,help)
    import nargs
    help.addArgument('HYPRE', '-with-hypre-gpu-arch=<string>',  nargs.ArgString(None, 0, 'Value passed to hypre\'s --with-gpu-arch= configure option'))
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.openmp        = framework.require('config.packages.openmp',self)
    self.cxxlibs       = framework.require('config.packages.cxxlibs',self)
    self.blasLapack    = framework.require('config.packages.BlasLapack',self)
    self.mpi           = framework.require('config.packages.MPI',self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.cuda          = framework.require('config.packages.cuda',self)
    self.hip           = framework.require('config.packages.hip',self)
    self.openmp        = framework.require('config.packages.openmp',self)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.scalar        = framework.require('PETSc.options.scalarTypes',self)
    self.deps          = [self.mpi,self.blasLapack,self.cxxlibs,self.mathlib]
    self.odeps         = [self.cuda,self.hip,self.openmp]
    if self.setCompilers.isCrayKNL(None,self.log):
      self.installwithbatch = 0

  def formGNUConfigureArgs(self):
    self.packageDir = os.path.join(self.packageDir,'src')
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if not hasattr(self.compilers, 'CXX'):
      raise RuntimeError('Error: Hypre requires C++ compiler. None specified')
    if not hasattr(self.compilers, 'FC'):
      args.append('--disable-fortran')
    if self.mpi.include:
      # just use the first dir - and assume the subsequent one isn't necessary [relavant only on AIX?]
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

    cucc = ''
    devflags = ''
    hipbuild = False
    cudabuild = False
    hasharch = 'with-gpu-arch' in  args
    if self.hip.found:
      hipbuild = True
      args.append('--with-hip')
      if not hasharch:
        if not 'with-hypre-gpu-arch' in self.framework.clArgDB:
          args.append('--with-gpu-arch=gfx908') # default
        else:
          args.append('--with-gpu-arch='+self.argDB['with-hypre-gpu-arch'])
      self.pushLanguage('HIP')
      cucc = self.getCompiler()
      devflags += ' -x hip -std=c++14 '
      devflags += self.getCompilerFlags() + ' ' + self.setCompilers.HIPPPFLAGS + ' ' + self.mpi.includepaths
      devflags = devflags.replace('-fvisibility=hidden','')
      self.popLanguage()
    elif self.cuda.found:
      cudabuild = True
      args.append('CUDA_HOME="'+self.cuda.cudaDir+'"')
      args.append('--with-cuda')
      if not hasharch:
        if not 'with-hypre-gpu-arch' in self.framework.clArgDB:
          if hasattr(self.cuda,'gencodearch'):
            args.append('--with-gpu-arch=' + self.cuda.gencodearch)
          else:
            args.append('--with-gpu-arch=70') # default
        else:
          args.append('--with-gpu-arch='+self.argDB['with-hypre-gpu-arch'])
      self.pushLanguage('CUDA')
      cucc = self.getCompiler()
      devflags += ' -expt-extended-lambda -std=c++11 --x cu '
      devflags += self.getCompilerFlags() + ' ' + self.setCompilers.CUDAPPFLAGS + ' ' + self.mpi.includepaths
      self.popLanguage()
    elif self.openmp.found:
      args.append('--with-openmp')
      self.usesopenmp = 'yes'
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
    args.append('--without-fei')
    args.append('--without-superlu')

    if self.getDefaultIndexSize() == 64:
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
    if cudabuild:
      args = [arg.replace('-std=gnu++14','-std=c++11') for arg in args]
      args = [arg.replace('-std=c++14','-std=c++11') for arg in args]
      args = [arg.replace('-std=c++17','-std=c++11') for arg in args]
    if hipbuild:
      args = [arg.replace('-std=c++17','-std=c++14') for arg in args]

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
