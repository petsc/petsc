import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '8.0.0'
    self.versionname      = 'STRUMPACK_VERSION_MAJOR.STRUMPACK_VERSION_MINOR.STRUMPACK_VERSION_PATCH'
    self.versioninclude   = 'StrumpackConfig.hpp'
    self.gitcommit        = 'v'+self.version
    self.download         = ['git://https://github.com/pghysels/STRUMPACK','https://github.com/pghysels/STRUMPACK/archive/v'+self.version+'.tar.gz']
    self.functionsCxx     = [1,'STRUMPACK_init','']
    self.includes         = ['StrumpackSparseSolver.h']
    self.liblist          = [['libstrumpack.a']]
    self.buildLanguages   = ['Cxx','FC']
    self.hastests         = 1
    self.minCmakeVersion  = (3,21,0)
    return

  def __str__(self):
    output  = config.package.CMakePackage.__str__(self)
    if hasattr(self,'system'): output += '  Backend: '+self.system+'\n'
    return output

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.scalapack      = framework.require('config.packages.ScaLAPACK',self)
    self.metis          = framework.require('config.packages.METIS',self)
    self.parmetis       = framework.require('config.packages.ParMETIS',self)
    self.ptscotch       = framework.require('config.packages.PTSCOTCH',self)
    self.zfp            = framework.require('config.packages.zfp',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.openmp         = framework.require('config.packages.OpenMP',self)
    self.cuda           = framework.require('config.packages.CUDA',self)
    self.hip            = framework.require('config.packages.HIP',self)
    self.slate          = framework.require('config.packages.slate',self)
    self.magma          = framework.require('config.packages.MAGMA',self)
    self.butterflypack  = framework.require('config.packages.butterflypack',self)
    self.deps           = [self.mpi,self.blasLapack,self.scalapack,self.metis]
    self.odeps          = [self.parmetis,self.ptscotch,self.openmp,self.cuda,self.hip,self.magma,self.slate,self.zfp,self.butterflypack]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    # ref. https://portal.nersc.gov/project/sparse/strumpack/master/installation.html
    args.append('-DTPL_BLAS_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DTPL_LAPACK_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DTPL_SCALAPACK_LIBRARIES="'+self.libraries.toString(self.scalapack.lib)+'"')

    args.append('-DTPL_METIS_LIBRARIES="'+self.libraries.toString(self.metis.lib)+'"')
    args.append('-DTPL_METIS_INCLUDE_DIRS="'+self.headers.toStringNoDupes(self.metis.include)[2:]+'"')

    if self.parmetis.found:
      args.append('-DTPL_ENABLE_PARMETIS=ON')
      args.append('-DTPL_PARMETIS_LIBRARIES="'+self.libraries.toString(self.parmetis.lib)+'"')
      args.append('-DTPL_PARMETIS_INCLUDE_DIRS="'+self.headers.toStringNoDupes(self.parmetis.include)[2:]+'"')
    else:
      args.append('-DTPL_ENABLE_PARMETIS=OFF')

    if self.ptscotch.found:
      args.append('-DTPL_ENABLE_SCOTCH=ON')
      args.append('-DTPL_SCOTCH_LIBRARIES="'+self.libraries.toString(self.ptscotch.lib)+'"')
      args.append('-DTPL_SCOTCH_INCLUDE_DIRS="'+self.headers.toStringNoDupes(self.ptscotch.include)[2:]+'"')
      args.append('-DTPL_ENABLE_PTSCOTCH=ON')
      args.append('-DTPL_PTSCOTCH_LIBRARIES="'+self.libraries.toString(self.ptscotch.lib)+'"')
      args.append('-DTPL_PTSCOTCH_INCLUDE_DIRS="'+self.headers.toStringNoDupes(self.ptscotch.include)[2:]+'"')
    else:
      args.append('-DTPL_ENABLE_SCOTCH=OFF')
      args.append('-DTPL_ENABLE_PTSCOTCH=OFF')

    if self.openmp.found:
      args.append('-DSTRUMPACK_USE_OPENMP=ON')
    else:
      args.append('-DSTRUMPACK_USE_OPENMP=OFF')

    if self.zfp.found:
      args.append('-DSTRUMPACK_USE_ZFP=ON')
    else:
      args.append('-DSTRUMPACK_USE_ZFP=OFF')

    if self.butterflypack.found:
      args.append('-DSTRUMPACK_USE_BPACK=ON')
    else:
      args.append('-DSTRUMPACK_USE_BPACK=OFF')

    # https://portal.nersc.gov/project/sparse/strumpack/master/GPU_Support.html
    if self.cuda.found:
      args.append('-DSTRUMPACK_USE_CUDA=ON')
      args.extend(self.cuda.getCmakeCUDAArchFlag())
    elif self.hip.found:
      args.append('-DSTRUMPACK_USE_HIP=ON')
      # Not using -DHIP_HIPCC_FLAGS=--amdgpu-target=gfx906 as mentioned in the doc, because we prefer standardized cmake options
      args.append('-DCMAKE_HIP_ARCHITECTURES="'+self.hip.hipArch+'"') # cmake supports format like "gfx801;gfx900"

    if self.cuda.found or self.hip.found:
      if self.slate.found:
        args.append('-DSTRUMPACK_USE_SLATE=ON')
      else:
        args.append('-DSTRUMPACK_USE_SLATE=OFF')
        self.logPrintWarning('Strumpack requires SLATE as a GPU enabled ScaLAPACK alternative, reconfigure with --download-slate')

    return args

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      # Strumpack requires dlapmr() LAPACK routine
      if not self.blasLapack.checkForRoutine('dlapmr'):
        raise RuntimeError('Strumpack requires the LAPACK routine dlapmr(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it\nTry using --download-fblaslapack=1 option \nIf you are using vecLib on OSX, it does not contain this function.')
      self.log.write('Found dlapmr() in Lapack library as needed by Strumpack\n')
    return
