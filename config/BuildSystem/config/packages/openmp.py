import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions         = []
    self.includes          = ['omp.h']
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.function     = framework.require('config.functions',self)
    self.mpi          = framework.require('config.packages.MPI',self)
    self.pthread      = framework.require('config.packages.pthread',self)
    self.hwloc        = framework.require('config.packages.hwloc',self)
    self.odeps        = [self.mpi,self.pthread,self.hwloc]
    return

  def configureLibrary(self):
    ''' Checks for OpenMP compiler flags'''
    ''' Note it may be different for the C, C++, and FC compilers'''
    ''' Needs to check if OpenMP actually exists and works '''
    self.found = 0
    oflags = ["-fopenmp", # Gnu
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
