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
    self.mpi          = framework.require('config.packages.MPI',self)
    self.pthread      = framework.require('config.packages.pthread',self)
    self.hwloc        = framework.require('config.packages.hwloc',self)
    self.odeps        = [self.mpi,self.pthread,self.hwloc]
    return

  def configureLibrary(self):
    ''' Checks for -fopenmp compiler flag'''
    ''' Needs to check if OpenMP actually exists and works '''
    self.found = 0
    self.setCompilers.pushLanguage('C')
    #
    for flag in ["-fopenmp", # Gnu
                 "-qsmp=omp",# IBM XL C/C++
                 "-h omp",   # Cray. Must come after XL because XL interprets this option as meaning "-soname omp"
                 "-mp",      # Portland Group
                 "-Qopenmp", # Intel windows
                 "-openmp",  # Intel
                 "-xopenmp", # Sun
                 "+Oopenmp", # HP
                 "/openmp"   # Microsoft Visual Studio
                 #" ",        # Empty, if compiler automatically accepts openmp
                 ]:
      # here it should actually check if the OpenMP pragmas work here.
      if self.setCompilers.checkCompilerFlag(flag):
        ompflag = flag
        self.found = 1
        # Since clang on Apple doesn't work OpenMP, need to provide flag to GNU compiler that does the linking
        if self.setCompilers.isDarwin(self.log):
          self.setCompilers.LDFLAGS += ' '+ompflag
        break
    if not self.found:
      raise RuntimeError('Compiler has no support for OpenMP')
    self.setCompilers.addCompilerFlag(ompflag)
    self.setCompilers.popLanguage()
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      self.setCompilers.addCompilerFlag(ompflag)
      self.setCompilers.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.setCompilers.pushLanguage('Cxx')
      self.setCompilers.addCompilerFlag(ompflag)
      self.setCompilers.popLanguage()
    # register package since config.package.Package.configureLibrary(self) will not work since there is no library to find
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)
    config.package.Package.configureLibrary(self)
    # this is different from HAVE_OPENMP. HAVE_OPENMP_SUPPORT checks if we have facilities to support
    # running PETSc in flat-MPI mode and third party libraries in MPI+OpenMP hybrid mode
    if self.mpi.found and self.mpi.support_mpi3_shm and self.pthread.found and self.hwloc.found:
      self.addDefine('HAVE_OPENMP_SUPPORT', 1)
