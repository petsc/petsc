import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions         = []
    self.includes          = ['omp.h']
    self.liblist           = []
    self.complex           = 1   # 0 means cannot use complex
    self.double            = 0   # 1 means requires double precision 
    self.requires32bitint  = 0;  # 1 means that the package will not work with 64 bit integers
    self.worksonWindows    = 1  # 1 means that package can be used on Microsof Windows
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.pthreadclasses = framework.require('PETSc.packages.pthreadclasses',self)
    self.deps = [self.pthreadclasses]
    return

  def configureLibrary(self):
    ''' Checks for -fopenmp compiler flag'''
    ''' Needs to check if OpenMP actually exists and works '''
    self.setCompilers.pushLanguage('C')
    # 
    for flag in ["-fopenmp", # Gnu
                 "-qsmp=omp",# IBM XL C/C++
                 "-h omp",   # Cray. Must come after XL because XL interprets this option as meaning "-soname omp"
                 "-mp",      # Portland Group
                 "-Qopenmp", # Intel windows
                 "-openmp",  # Intel
                 " ",        # Empty, if compiler automatically accepts openmp
                 "-xopenmp", # Sun
                 "+Oopenmp", # HP
                 "/openmp"   # Microsoft Visual Studio
                 ]:
      if self.setCompilers.checkCompilerFlag(flag):
        ompflag = flag
        break
    self.setCompilers.addCompilerFlag(ompflag)
    if self.setCompilers.checkLinkerFlag(ompflag):
      self.setCompilers.addLinkerFlag(ompflag)
    self.setCompilers.popLanguage()
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      self.setCompilers.addCompilerFlag(ompflag)
      self.setCompilers.popLanguage()
    if self.languages.clanguage == 'Cxx':
      self.setCompilers.pushLanguage('Cxx')
      self.setCompilers.addCompilerFlag(ompflag)
      if self.setCompilers.checkLinkerFlag(ompflag):
        self.setCompilers.addLinkerFlag(ompflag)
      self.setCompilers.popLanguage()
    # OpenMP threadprivate variables are not supported on all platforms (for e.g on MacOS).
    # Hence forcing to configure additionally with --with-pthreadclasses so that pthread
    # routines pthread_get/setspecific() can be used instead.
    if not self.checkCompile('#include <omp.h>\nint a;\n#pragma omp threadprivate(a)\n','') and not self.pthreadclasses.found:
      raise RuntimeError('OpenMP threadprivate variables not found. Configure additionally with --with-pthreadclasses=1')
    # register package since PETSc.package.NewPackage.configureLibrary(self) will not work since there is no library to find
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)
