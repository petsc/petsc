import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    import os
    config.package.CMakePackage.__init__(self, framework)
    self.download          = ['git://https://github.com/meshadaptation/pragmatic.git']
    self.gitcommit         = '1dfe81ff5f34b16c15ee61adaaa7f4974e5b5135'
    self.functions         = ['pragmatic_2d_init']
    self.includes          = ['pragmatic/pragmatic.h']
    self.liblist           = [['libpragmatic.a']]
    self.requirescxx11     = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.scalartypes     = framework.require('PETSc.options.scalarTypes',self)
    self.indexTypes      = framework.require('PETSc.options.indexTypes', self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.metis           = framework.require('config.packages.metis', self)
    self.eigen           = framework.require('config.packages.eigen', self)
    self.mathlib         = framework.require('config.packages.mathlib',self)
    self.deps            = [self.mpi, self.metis, self.eigen, self.mathlib]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DMETIS_DIR='+self.metis.directory)
    args.append('-DENABLE_VTK=OFF')
    args.append('-DENABLE_OPENMP=OFF')
    args.append('-DEIGEN_INCLUDE_DIR='+self.eigen.include[0])

    args.append('-DMPI_C_INCLUDE_PATH:STRING=""')
    args.append('-DMPI_C_COMPILE_FLAGS:STRING=""')
    args.append('-DMPI_C_LINK_FLAGS:STRING=""')
    args.append('-DMPI_C_LIBRARIES:STRING=""')
    args.append('-DMPI_CXX_INCLUDE_PATH:STRING=""')
    args.append('-DMPI_CXX_COMPILE_FLAGS:STRING=""')
    args.append('-DMPI_CXX_LINK_FLAGS:STRING=""')
    args.append('-DMPI_CXX_LIBRARIES:STRING=""')

    if self.checkSharedLibrariesEnabled():
      args.append('-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON')
    if self.indexTypes.integerSize == 64:
      raise RuntimeError('Pragmatic cannot be built with 64-bit integers')
    if self.scalartypes.precision == 'single':
      raise RuntimeError('Pragmatic cannot be built with single precision')
    elif self.scalartypes.precision == '__float128':
      raise RuntimeError('Pragmatic cannot be built with quad precision')
    return args
