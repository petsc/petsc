import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions = ['cublasInit']
    self.includes  = ['cublas.h']
    self.liblist   = [['libcublas.a','libcudart.a']]
    self.double    = 0   # 1 means requires double precision 
    self.cxx       = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers',self)
    self.headers      = framework.require('config.headers',self)
    self.scalartypes  = framework.require('PETSc.utilities.scalarTypes', self)        
    self.languages    = framework.require('PETSc.utilities.languages',   self)
    self.cusp         = framework.require('config.packages.cusp',        self)
    self.thrust       = framework.require('config.packages.thrust',      self)
    return

  def getSearchDirectories(self):
    yield os.path.join('/usr','local','cuda')
    return

  def configureTypes(self):
    if self.scalartypes.scalartype == 'complex':
      raise RuntimeError('Must use real numbers with CUDA') 
    if not self.scalartypes.precision in ['double', 'single']:
      raise RuntimeError('Must use either single or double precision with CUDA') 
    else:
      self.pushLanguage('CUDA')
      self.setCompilers.addCompilerFlag('-m64')
      if self.scalartypes.precision == 'double':
        self.setCompilers.addCompilerFlag('-arch sm_13')
      self.popLanguage()
    return

  def configureLibrary(self):
    PETSc.package.NewPackage.configureLibrary(self)
    if not self.cusp.found or not self.thrust.found:
      raise RuntimeError('PETSc CUDA support requires the CUSP and THRUST packages\nRerun configure using --with-cusp-dir and --with-thrust-dir')
    if self.languages.clanguage == 'C':
      self.addDefine('CUDA_EXTERN_C_BEGIN','extern "C" {')
      self.addDefine('CUDA_EXTERN_C_END','}')
    else:
      self.addDefine('CUDA_EXTERN_C_BEGIN',' ')
      self.addDefine('CUDA_EXTERN_C_END',' ')
    self.configureTypes()
    #self.include = self.include+[os.path.join('/usr','local','cuda')]
    #if hasattr(self.compilers, 'CXX') and self.languages.clanguage == 'C':
    #  self.setCompilers.pushLanguage('Cxx')
    #  cxx_linker = self.setCompilers.getLinker()
    #  self.setCompilers.popLanguage()
    #  self.addMakeMacro('CLINKER', cxx_linker)
    return
