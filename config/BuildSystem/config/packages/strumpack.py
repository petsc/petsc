import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.download         = ['http://portal.nersc.gov/project/sparse/strumpack/STRUMPACK-sparse-1.0.0.tar.gz']
    self.functions        = ['STRUMPACK_init']
    self.includes         = ['StrumpackSparseSolver.h']
    self.liblist          = [['libstrumpack_sparse.a']]
    self.cxx              = 1
    self.fc               = 1
    self.requirescxx11    = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.scalapack      = framework.require('config.packages.scalapack',self)
    self.metis          = framework.require('config.packages.metis',self)
    self.parmetis       = framework.require('config.packages.parmetis',self)
    self.ptscotch       = framework.require('config.packages.PTScotch',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.openmp         = framework.require('config.packages.openmp',self)
    self.deps           = [self.mpi,self.openmp,self.blasLapack,self.scalapack,self.parmetis,self.metis,self.ptscotch]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DCMAKE_CXX_FLAGS="' \
                + self.headers.toStringNoDupes(self.ptscotch.include) + ' ' \
                + self.headers.toStringNoDupes(self.metis.include) + ' ' \
                + self.headers.toStringNoDupes(self.parmetis.include) + '"')
    args.append('-DCMAKE_EXE_LINKER_FLAGS="' \
                + self.libraries.toString(self.blasLapack.dlib) + ' ' \
                + self.libraries.toString(self.scalapack.lib) + ' ' \
                + self.libraries.toStringNoDupes(self.ptscotch.lib) + ' ' \
                + self.libraries.toStringNoDupes(self.metis.lib) + ' ' \
                + self.libraries.toStringNoDupes(self.parmetis.lib) + '"')

    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=Debug')
    else:
      args.append('-DCMAKE_BUILD_TYPE=Release')

    # building with shared libs results in things like:
    #  /usr/bin/ld: ex5: hidden symbol `SCOTCH_dgraphOrderCompute' in
    #        /home/pieterg/workspace/petsc/arch-linux2-c-debug/lib/libptscotch.a(library_dgraph_order.o)
    #          is referenced by DSO
    #if not self.checkSharedLibrariesEnabled():
    args.append('-DBUILD_SHARED_LIBS=off')

    self.framework.pushLanguage('C')
    args.append('-DMPI_C_COMPILER="' + self.framework.getCompiler() + '"')
    self.framework.popLanguage()

    self.framework.pushLanguage('Cxx')
    args.append('-DMPI_CXX_COMPILER="' + self.framework.getCompiler() + '"')
    self.framework.popLanguage()

    self.framework.pushLanguage('FC')
    args.append('-DMPI_Fortran_COMPILER="' + self.framework.getCompiler() + '"')
    self.framework.popLanguage()

    args.append('-DCMAKE_INSTALL_NAME_DIR:STRING="'+os.path.join(self.installDir,self.libdir)+'"')

    return args
