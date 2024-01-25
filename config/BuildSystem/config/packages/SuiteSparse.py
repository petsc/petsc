import os

import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.minversion        = '5.6.0'
    self.version           = '7.6.0'
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
    self.hastests          = 1
    self.buildLanguages    = ['Cxx']
    self.hastestsdatafiles = 1
    self.precisions        = ['double']
    self.minCmakeVersion   = (3,22,0)
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mathlib    = framework.require('config.packages.mathlib',self)
    self.deps       = [self.blasLapack,self.mathlib]
    self.cuda       = framework.require('config.packages.cuda',self)
    self.openmp     = framework.require('config.packages.openmp',self)
    self.odeps      = [self.openmp,self.cuda]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DSUITESPARSE_ENABLE_PROJECTS="amd;cholmod;klu;umfpack;spqr"')
    args.append('-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON')

    args.append('-DBLA_VENDOR:STRING=Generic')
    args.append('-DBLAS_LIBRARIES:STRING="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DLAPACK_LIBRARIES:STRING=""')

    args.append('-DSUITESPARSE_USE_CUDA:BOOL='+('ON' if self.cuda.found else 'OFF'))
    args.append('-DSUITESPARSE_USE_OPENMP:BOOL='+('ON' if self.openmp.found else 'OFF'))
    args.append('-DSUITESPARSE_USE_64BIT_BLAS:BOOL='+('ON' if self.blasLapack.has64bitindices else 'OFF'))

    return args
