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
    self.setCompilers.pushLanguage('C')

    self.found = 0
    for flag in oflags:
      if self.setCompilers.checkCompilerFlag(flag):
        ompflag = flag
        self.found = 1
        # OpenMP compile flag is also needed at link time
        self.setCompilers.LDFLAGS += ' '+ompflag
        oldFlags = self.compilers.CPPFLAGS
        self.compilers.CPPFLAGS += ompflag
        try:
          output,err,status  = self.preprocess('#if defined(_OPENMP)\nopmv=_OPENMP\n#else\n#error "No _OPENMP macro, something is wrong with the OpenMP install"\n#endif')
        except:
          raise RuntimeError('Unable to run preprocessor to determine if OpenMP compile flag worked')
        loutput = output.split('\n')
        for i in loutput:
          if i.startswith('opmv='):
            self.foundversion = i[5:]
            self.ompflag = ompflag
            break
          if i.startswith('opmv ='):
            self.foundversion = i[6:]
            self.ompflag = ompflag
            break
        self.compilers.CPPFLAGS = oldFlags
        break
    if not self.found:
      raise RuntimeError('C Compiler has no support for OpenMP')
    self.setCompilers.addCompilerFlag(ompflag)
    self.setCompilers.popLanguage()

    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      self.found = 0
      for flag in oflags:
        if self.setCompilers.checkCompilerFlag(flag):
          ompflag = flag
          self.found = 1
          oldFlags = self.compilers.CPPFLAGS
          self.compilers.CPPFLAGS += ompflag
          try:
            output,err,status  = self.preprocess('#if !defined(_OPENMP)\n#error "No _OPENMP macro, something is wrong with the OpenMP install"\n#endif')
          except:
            raise RuntimeError('Unable to run preprocessor to determine if OpenMP compile flag worked')
          self.compilers.CPPFLAGS = oldFlags
          break
      if not self.found:
        raise RuntimeError('Fortran Compiler has no support for OpenMP')
      self.setCompilers.addCompilerFlag(ompflag)
      self.setCompilers.popLanguage()

    if hasattr(self.compilers, 'CXX'):
      self.setCompilers.pushLanguage('Cxx')
      self.found = 0
      for flag in oflags:
        if self.setCompilers.checkCompilerFlag(flag):
          ompflag = flag
          self.found = 1
          oldFlags = self.compilers.CPPFLAGS
          self.compilers.CPPFLAGS += ompflag
          try:
            output,err,status  = self.preprocess('#if !defined(_OPENMP)\n#error "No _OPENMP macro, something is wrong with the OpenMP install"\n#endif')
          except:
            raise RuntimeError('Unable to run preprocessor to determine if OpenMP compile flag worked')
          self.compilers.CPPFLAGS = oldFlags
          break
      if not self.found:
        raise RuntimeError('CXX Compiler has no support for OpenMP')
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
