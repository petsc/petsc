if __name__ == '__main__':
 import sys
 import os
 sys.path.insert(0, os.path.abspath('config'))
 import configure
 configure_options = [
  '--CC_LINKER_FLAGS=msvcrt.lib',
  '--CXX_LINKER_FLAGS=msvcrt.lib',
  '--doCleanup=1',
  '--useThreads=0',
  '--with-debugging=1',
  '--with-64-bit-indices=1',
  '--known-64-bit-blas-indices=1',
  '--with-cc=win32fe cl',
  '--with-cxx=win32fe cl',
  '--CFLAGS=-DMKL_ILP64 -MD',
  '--CPPFLAGS=-DMKL_ILP64 -MD',
  '--with-fc=win32fe ifort',
  '--FFLAGS=-4I8',
  '--with-cuda=0',
  '--with-mpi=1',
  '--with-mpi-include=/LIBS64/ICS2015SP0/Intel/mpi/include',
  '--with-mpi-lib=[/LIBS64/ICS2015SP0/Intel/mpi/lib/impi.lib,/LIBS64/ICS2015SP0/Intel/mpi/lib/impicxx.lib]',
  '--with-mpiexec=/LIBS64/ICS2015SP0/Intel/mpi/bin/mpiexec.exe',
  '--with-blas-include=/LIBS64/ICS2015SP0/Intel/mkl/include',
  '--with-blas-lib=[-L/LIBS64/ICS2015SP0/Intel/mkl/lib/intel64,mkl_scalapack_ilp64.lib,mkl_cdft_core.lib,mkl_intel_ilp64.lib,mkl_core.lib,mkl_intel_thread.lib,mkl_blacs_intelmpi_ilp64.lib,libiomp5md.lib]',
  '--with-lapack-include=/LIBS64/ICS2015SP0/Intel/mkl/include',
  '--with-lapack-lib=[-L/LIBS64/ICS2015SP0/Intel/mkl/lib/intel64,mkl_scalapack_ilp64.lib,mkl_cdft_core.lib,mkl_intel_ilp64.lib,mkl_core.lib,mkl_intel_thread.lib,mkl_blacs_intelmpi_ilp64.lib,libiomp5md.lib]',
  '--with-blacs-include=/LIBS64/ICS2015SP0/Intel/mkl/include',
  '--with-blacs-lib=[-L/LIBS64/ICS2015SP0/Intel/mkl/lib/intel64,mkl_scalapack_ilp64.lib,mkl_cdft_core.lib,mkl_intel_ilp64.lib,mkl_core.lib,mkl_intel_thread.lib,mkl_blacs_intelmpi_ilp64.lib,libiomp5md.lib]',
  '--with-scalapack-include=/LIBS64/ICS2015SP0/Intel/mkl/include',
  '--with-scalapack-lib=[-L/LIBS64/ICS2015SP0/Intel/mkl/lib/intel64,mkl_scalapack_ilp64.lib,mkl_cdft_core.lib,mkl_intel_ilp64.lib,mkl_core.lib,mkl_intel_thread.lib,mkl_blacs_intelmpi_ilp64.lib,libiomp5md.lib]',
  '--with-mkl_pardiso-dir=/LIBS64/ICS2015SP0/Intel/mkl',
  '--with-mkl_cpardiso-dir=/LIBS64/ICS2015SP0/Intel/mkl',
  '--pc-type lu',
  '--pc_factor_mat_solver_type mkl_pardiso',
  '--pc_factor_mat_solver_type mkl_cpardiso',
  'PETSC_ARCH=arch-ms-msvc2012-intelmpi-cudano-nomumps-cpardiso-indexes64-mklilp64-debug'
 ]
 configure.petsc_configure(configure_options)
