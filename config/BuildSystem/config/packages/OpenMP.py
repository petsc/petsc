import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions         = []
    self.includes          = ['omp.h']
    return

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    help.addArgument('OpenMP', '--with-openmp-kernels=<true,false>',  nargs.ArgBool(None, 0, 'PETSc\'s numerical kernels will use OpenMP threads'))
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.function     = framework.require('config.functions',self)
    self.mpi          = framework.require('config.packages.MPI',self)
    self.pthread      = framework.require('config.packages.pthread',self)
    self.hwloc        = framework.require('config.packages.hwloc',self)
    self.cuda         = framework.require('config.packages.CUDA',self)
    self.hip          = framework.require('config.packages.HIP',self)
    self.odeps        = [self.mpi,self.pthread,self.hwloc,self.cuda,self.hip]
    return

  def configureLibrary(self):
    ''' Checks for OpenMP compiler flags'''
    ''' Note it may be different for the C, C++, and FC compilers'''
    ''' Needs to check if OpenMP actually exists and works '''
    oflags = ["-qopenmp", # Intel (must come before -fopenmp for icx)
              "-fopenmp", # Gnu
              "-qsmp=omp",# IBM XL C/C++
              "-h omp",   # Cray. Must come after XL because XL interprets this option as meaning "-soname omp"
              "-mp",      # Portland Group
              "-Qopenmp", # Intel windows
              "-openmp",  # Intel
              "-xopenmp", # Sun
              "+Oopenmp", # HP
              "/openmp"   # Microsoft Visual Studio
              ]

    # No ('CUDA','CUDAC') since cuda host code is compiled by host CXX
    foundversion = False
    linkers = []
    for language, compiler in [('C','CC'),('Cxx','CXX'),('FC','FC'),('HIP','HIPC'),('SYCL','SYCLC')]:
      if hasattr(self.compilers, compiler):
        self.setCompilers.pushLanguage(language)
        self.found = 0
        for flag in oflags:
          if self.setCompilers.checkCompilerFlag(flag):
            ompflag = flag
            self.found = 1
            # Flag is sometimes needed at preprocessor time so put it there and NOT in compiler flags
            flagsName = self.getPreprocessorFlagsName(language)
            oldFlags = getattr(self.setCompilers, flagsName)
            setattr(self.setCompilers, flagsName, oldFlags+' '+ompflag)
            try:
              output,err,status  = self.preprocess('#if defined(_OPENMP)\nompv=_OPENMP\n#else\n#error "No _OPENMP macro for '+compiler+', something is wrong with the OpenMP install"\n#endif')
            except:
              raise RuntimeError('Unable to run preprocessor to determine if OpenMP compile flag worked')
            if not foundversion and language in ['C','Cxx']:
              loutput = output.split('\n')
              for i in loutput:
                if i.startswith('ompv='):
                  self.foundversion = i[5:]
                  self.ompflag = ompflag
                  foundversion = True
                  break
            # OpenMP compile flag is also needed at link time but preprocessor flags not passed to linker
            linker = self.setCompilers.getLinkerFlagsArg()
            if linker+' '+ompflag not in linkers:
              self.setCompilers.addLinkerFlag(ompflag)
              linkers.append(linker+' '+ompflag)
            break
        if not self.found:
          raise RuntimeError(compiler + ' Compiler has no support for OpenMP')
        self.setCompilers.popLanguage()

    # register package since config.package.Package.configureLibrary(self) will not work since there is no library to find
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)
    config.package.Package.configureLibrary(self)
    # this is different from HAVE_OPENMP. HAVE_OPENMP_SUPPORT checks if we have facilities to support
    # running PETSc in flat-MPI mode and third party libraries in MPI+OpenMP hybrid mode
    if self.mpi.found and self.mpi.support_mpi3_shm and self.pthread.found and self.hwloc.found:
      #  Apple pthread does not provide this functionality
      if self.function.check('pthread_barrier_init', libraries = 'pthread'):
        self.addDefine('HAVE_OPENMP_SUPPORT', 1)

    if self.framework.argDB['with-openmp-kernels']:
      self.addDefine('USE_OPENMP_KERNELS', 1)

    self.configureOpenMPTarget()

  def configureOpenMPTarget(self):
    '''Automatically enable OpenMP GPU target offload for each host compiler that supports it.

       Since users might use a Fortran compiler supporting OpenMP target offload along with
       regular C/C++ compilers, we do this check per host language and define
       PETSC_HAVE_OPENMP_TARGET_OFFLOAD_{CC, CXX, FC} accordingly.
    '''
    if self.cuda.found:  candidates = ['-foffload=nvptx-none', '-mp=gpu', '-fopenmp-targets=nvptx64-nvidia-cuda'] # flags for [GNU, NVIDIA, Clang] compilers
    elif self.hip.found: candidates = ['-foffload=amdgcn-amdhsa', '-fopenmp-targets=amdgcn-amd-amdhsa'] # flags for [GNU, Clang] compilers
    else: return # TODO: support SYCL.

    # A minimal OpenMP target region per language
    ctest = ('#include <omp.h>\n', 'int x = 0;\n#pragma omp target map(tofrom:x)\n  x = 1;\n  (void)x;\n')
    ftest = ('', 'integer :: x\n      x = 0\n!$omp target map(tofrom:x)\n      x = 1\n!$omp end target')

    for language, compiler, (includes, body) in [('C', 'CC', ctest), ('Cxx', 'CXX', ctest), ('FC', 'FC', ftest)]:
      if not hasattr(self.compilers, compiler): continue
      self.setCompilers.pushLanguage(language)
      cflagsArg  = self.setCompilers.getCompilerFlagsArg(0)
      ldflagsArg = self.setCompilers.getLinkerFlagsArg()
      for flag in candidates:
        oldc = getattr(self.setCompilers, cflagsArg)
        oldl = getattr(self.setCompilers, ldflagsArg)
        setattr(self.setCompilers, cflagsArg, oldc + ' ' + flag)
        setattr(self.setCompilers, ldflagsArg, oldl + ' ' + flag)
        works = self.setCompilers.checkLink(includes, body)
        setattr(self.setCompilers, cflagsArg, oldc)
        setattr(self.setCompilers, ldflagsArg, oldl)
        if works:
          # Record the flag only in this language's compiler flags (CFLAGS/CXXFLAGS/FFLAGS); those
          # already flow into the corresponding language's link line. Do not add it to the linker
          # flags: getLinkerFlagsArg() is LDFLAGS, which is shared by C, C++, and Fortran, so a flag
          # accepted by one compiler would leak onto the others' link lines and break them.
          self.setCompilers.insertCompilerFlag(flag, 0)
          self.addDefine('HAVE_OPENMP_TARGET_OFFLOAD_' + compiler, 1)
          self.logPrintBox('Enabled OpenMP target offload for ' + compiler + ': ' + flag)
          break
      self.setCompilers.popLanguage()

  def alternateConfigureLibrary(self):
    if self.framework.argDB['with-openmp-kernels']:
      raise RuntimeError('--with-openmp-kernels also requires --with-openmp')