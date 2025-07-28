import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '6.4.0'
    self.minversion       = '6.4.0'
    self.gitcommit        = 'v'+self.version
    self.versionname      = 'PASTIX_MAJOR_VERSION.PASTIX_MEDIUM_VERSION.PASTIX_MINOR_VERSION'
    self.download         = ['git://https://gitlab.inria.fr/solverstack/pastix.git',
                             'https://files.inria.fr/pastix/releases/v6/pastix-'+self.version+'.tar.gz',
                             'https://web.cels.anl.gov/projects/petsc/download/externalpackages/pastix-'+self.version+'.tar.gz']
    self.liblist          = [['libpastix.a']]
    self.functions        = ['pastixInit']
    self.includes         = ['pastix.h']
    self.precisions       = ['single','double']
    self.hastests         = 1
    self.hastestsdatafiles= 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.indexTypes      = framework.require('PETSc.options.indexTypes', self)
    self.blaslapack      = framework.require('config.packages.BlasLapack',self)
    self.metis           = framework.require('config.packages.METIS',self)
    self.ptscotch        = framework.require('config.packages.PTSCOTCH',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.pthread         = framework.require('config.packages.pthread',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.deps            = [self.blaslapack, self.pthread, self.hwloc]
    self.odeps           = [self.mpi, self.ptscotch, self.metis]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    if self.blaslapack.netliblapack.found and not self.blaslapack.netliblapack.cinterface:
      raise RuntimeError('If you use PaStiX with netlib-lapack you have to add the option --with-netlib-lapack-c-bindings')

    if not self.libraries.check(self.dlib, 'cblas_dgemm'):
      raise RuntimeError('PaStiX requires a BLAS library with cblas support. Suggest specifying MKL for ex. --with-blaslapack-dir=${MKLROOT}, or --download-netlib-lapack --with-netlib-lapack-c-bindings')

    if not self.libraries.check(self.dlib, 'LAPACKE_dlange'):
      raise RuntimeError('PaStiX requires a LAPACK library with LAPACKE support. Suggest specifying MKL for ex. --with-blaslapack-dir=${MKLROOT}, or --download-netlib-lapack --with-netlib-lapack-c-bindings')

    if not self.ptscotch.found and not self.metis.found:
      raise RuntimeError('PaStiX requires an ordering library: either METIS or SCOTCH')

    if hasattr(self.programs, 'pkg-config'):
      args.append('-DPKG_CONFIG_EXECUTABLE:STRING=\"{0}\"'.format(getattr(self.programs, 'pkg-config')))
    else:
      raise RuntimeError('PaStiX needs pkg-config installed')

    args.append('-DPASTIX_WITH_FORTRAN=OFF')
    args.append('-DPASTIX_LR_TESTINGS=OFF')

    if self.indexTypes.integerSize == 64:
      args.append("-DPASTIX_INT64=ON")
    else:
      args.append("-DPASTIX_INT64=OFF")

    if self.metis.found:
      args.append("-DPASTIX_ORDERING_METIS=ON")
    else :
      args.append("-DPASTIX_ORDERING_METIS=OFF")

    if self.ptscotch.found:
      args.append("-DPASTIX_ORDERING_SCOTCH=ON")
    else :
      args.append("-DPASTIX_ORDERING_SCOTCH=OFF")

    if self.mpi.found:
      args.append("-DPASTIX_WITH_MPI=ON")
    else:
      args.append("-DPASTIX_WITH_MPI=OFF")

    cmake_prefix_path = []
    cmake_include_path = []
    for dep in [self.blaslapack, self.metis, self.ptscotch, self.mpi, self.hwloc]:
      if dep.found:
        if dep.directory:
          if os.path.isdir(dep.directory):
            cmake_prefix_path.append(dep.directory)
        if dep.include:
          cmake_include_path.append(self.headers.toStringNoDupes(dep.include)[2:])

    if cmake_include_path:
        cmake_include_path = list(set(cmake_include_path))
        cmake_include_path = ';'.join(cmake_include_path)
        args.append("-DCMAKE_INCLUDE_PATH=\"{0}\"".format(cmake_include_path))

    if cmake_prefix_path:
      cmake_prefix_path = list(set(cmake_prefix_path))
      cmake_prefix_path = ';'.join(cmake_prefix_path)
      args.append("-DCMAKE_PREFIX_PATH=\"{0}\"".format(cmake_prefix_path))

    return args
