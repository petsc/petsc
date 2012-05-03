import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions         = []
    self.includes          = ['omp.h']
    self.liblist           = []
    self.complex           = 1   # 0 means cannot use complex
    self.lookforbydefault  = 1
    self.double            = 0   # 1 means requires double precision 
    self.requires32bitint  = 0;  # 1 means that the package will not work with 64 bit integers
    self.worksonWindows    = 1  # 1 means that package can be used on Microsof Windows
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = []
    return

  def getSearchDirectories(self):
    ''' libomp.a is in the usual place'''
    yield ''
    return

  def configureLibrary(self):
    ''' Checks for -fopenmp compiler flag'''
    PETSc.package.NewPackage.configureLibrary(self)
    self.setCompilers.pushLanguage('C')
    # 
    for flag in ["-fopenmp", # Gnu
                 "-h omp",   # Cray
                 "-mp",      # Portland Group
                 "-Qopenmp", # Intel windows
                 "-openmp",  # Intel
                 " ",        # Empty, if compiler automatically accepts openmp
                 "-xopenmp", # Sun
                 "+Oopenmp", # HP
                 "-qsmp",    # IBM XL C/c++
                 "/openmp"   # Microsoft Visual Studio
                 ]:
      if self.setCompilers.checkCompilerFlag(flag):
        ompflag = flag
        break
    self.setCompilers.addCompilerFlag(ompflag)
    self.setCompilers.popLanguage()
    self.setCompilers.pushLanguage('FC')
    self.setCompilers.addCompilerFlag(ompflag)
    self.setCompilers.popLanguage()
    if self.languages.clanguage == 'Cxx':
      self.setCompilers.pushLanguage('Cxx')
      self.setCompilers.addCompilerFlag(ompflag)
      self.setCompilers.popLanguage()
    self.addDefine('HAVE_OPENMP', '1')

# sets PETSC_HAVE_OPENMP and adds the relevant flag
# to the compile and link lines
