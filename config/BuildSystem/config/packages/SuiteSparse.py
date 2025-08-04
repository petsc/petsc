import os

import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.minversion        = '5.6.0'
    self.version           = '7.11.0'
    self.versioninclude    = 'SuiteSparse_config.h'
    self.versionname       = 'SUITESPARSE_MAIN_VERSION.SUITESPARSE_SUB_VERSION.SUITESPARSE_SUBSUB_VERSION'
    self.gitcommit         = 'v'+self.version
    self.download          = ['git://https://github.com/DrTimothyAldenDavis/SuiteSparse','https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/'+self.gitcommit+'.tar.gz']
    self.liblist           = [['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libamd.a'],
                              ['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libsuitesparseconfig.a'],
                              ['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libsuitesparseconfig.a','librt.a'],
                              ['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libmetis.a','libsuitesparseconfig.a'], # < 6.0.0
                              ['libspqr.a','libumfpack.a','libklu.a','libcholmod.a','libbtf.a','libccolamd.a','libcolamd.a','libcamd.a','libamd.a','libmetis.a','libsuitesparseconfig.a','librt.a']] # < 6.0.0
    self.functions         = ['umfpack_dl_wsolve','cholmod_l_solve','klu_l_solve','SuiteSparseQR_C_solve','amd_info']
    self.includes          = ['umfpack.h','cholmod.h','klu.h','SuiteSparseQR_C.h','amd.h']
    self.includedir        = [os.path.join('include', 'suitesparse'),'include']
    self.pkgname           = 'SPQR UMFPACK KLU CHOLMOD AMD'
    self.hastests          = 1
    self.buildLanguages    = ['Cxx']
    self.hastestsdatafiles = 1
    self.precisions        = ['double']
    self.minCmakeVersion   = (3,22,0)
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('SUITESPARSE', '-with-suitesparse-cuda=<bool>', nargs.ArgBool(None, 0, 'Compile SuiteSparse with CUDA enabled'))
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mathlib    = framework.require('config.packages.mathlib',self)
    self.deps       = [self.blasLapack,self.mathlib]
    self.cuda       = framework.require('config.packages.CUDA',self)
    self.openmp     = framework.require('config.packages.OpenMP',self)
    self.odeps      = [self.openmp,self.cuda]
    return

  def getSearchDirectories(self):
    '''Generate list of possible locations of SuiteSparse'''
    yield ''
    yield '/usr'
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DSUITESPARSE_ENABLE_PROJECTS="amd;cholmod;klu;umfpack;spqr"')

    args.append('-DBLA_VENDOR:STRING=Generic')
    args.append('-DBLAS_LIBRARIES:STRING="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DLAPACK_LIBRARIES:STRING=""')

    enablecuda = 0
    if self.cuda.found and self.openmp.found:
      enablecuda = 1
    if 'with-suitesparse-cuda' in self.framework.clArgDB:
      if self.argDB['with-suitesparse-cuda']:
        if not enablecuda:
          raise RuntimeError('SuiteSparse build with CUDA enabled requires --with-cuda=1 and --with-openmp=1')
      else:
        enablecuda = 0
    args.append('-DSUITESPARSE_USE_CUDA:BOOL='+('ON' if enablecuda else 'OFF'))
    args.append('-DSUITESPARSE_USE_OPENMP:BOOL='+('ON' if self.openmp.found else 'OFF'))
    args.append('-DSUITESPARSE_USE_64BIT_BLAS:BOOL='+('ON' if self.blasLapack.has64bitindices else 'OFF'))

    return args
