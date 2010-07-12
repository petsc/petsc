import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions = ['cublasInit']
    self.includes  = ['cublas.h']
    self.liblist   = [['libcublas.a']]
    self.double    = 0   # 1 means requires double precision 
    self.cxx       = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.headers        = framework.require('config.headers',self)
    self.scalartypes    = framework.require('PETSc.utilities.scalarTypes',     self)        
    self.languages      = framework.require('PETSc.utilities.languages',       self)

  def getSearchDirectories(self):
    yield os.path.join('/usr','local','cuda')

  def configureLibrary(self):
    PETSc.package.NewPackage.configureLibrary(self)
    self.setCompilers.pushLanguage('C++')
#    if not self.headers.checkInclude([os.path.join('/usr','local','cuda')],[os.path.join('thrust','advance.h')]):
#       raise RuntimeError('Cannot find thrust include files') 
#    if not self.headers.checkInclude([os.path.join('/usr','local','cuda')],[os.path.join('cusp','monitor.h')]):
#       raise RuntimeError('Cannot find cusp include files') 
    self.setCompilers.popLanguage()
    self.include = self.include+[os.path.join('/usr','local','cuda')]
    if self.scalartypes.precision == 'double':
      self.addMakeMacro('CUDACC','nvcc -m64 -arch sm_13')
    elif self.scalartypes.precision == 'single':
      self.addMakeMacro('CUDACC','nvcc -m64')
    else:
      raise RuntimeError('Must use either single or double precision with CUDA') 
    if self.scalartypes.scalartype == 'complex':
      raise RuntimeError('Must use real numbers with CUDA') 
    self.addMakeMacro('CLINKER','nvcc -m64')
    if self.languages.clanguage == 'C':
      self.addDefine('CUDA_EXTERN_C_BEGIN','extern "C" {')
      self.addDefine('CUDA_EXTERN_C_END','}')
    else:
      self.addDefine('CUDA_EXTERN_C_BEGIN','')
      self.addDefine('CUDA_EXTERN_C_END','')
    
# add checks that it is proper version o Cuda
