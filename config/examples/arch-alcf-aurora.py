#!/usr/bin/python3

# Follow instructions at https://docs.alcf.anl.gov/aurora/getting-started-on-aurora/#proxy
# to set up the proxy settings in your .bashrc and git with SSH protocol in your .ssh/config

# module use /soft/modulefiles
# module load cmake python autoconf
#
# Currently Loaded Modules:
#   1) gcc-runtime/13.3.0-ghotoln    (H)  10) yaksa/0.3-7ks5f26             (H)  19) libmd/1.0.4-q6tzwyj         (H)  28) abseil-cpp/20240722.0-ck5p27o (H)
#   2) gmp/6.3.0-mtokfaw             (H)  11) mpich/opt/develop-git.6037a7a      20) libbsd/0.12.2-wxndujc       (H)  29) c-ares/1.28.1-dqfje2b         (H)
#   3) mpfr/4.2.1-gkcdl5w            (H)  12) libfabric/1.22.0                   21) expat/2.6.4-7j6nhb6         (H)  30) protobuf/3.28.2
#   4) mpc/1.3.1-rdrlvsl             (H)  13) cray-pals/1.4.0                    22) sqlite/3.46.0-w5wc5lh       (H)  31) re2/2023-09-01-jnio6ml        (H)
#   5) gcc/13.3.0                         14) cray-libpals/1.4.0                 23) python/3.10.14                   32) grpc/1.66.1-yz5gmcn           (H)
#   6) oneapi/release/2025.0.5            15) bzip2/1.0.8                        24) berkeley-db/18.1.40-64t6wec (H)  33) nlohmann-json/3.11.3-hzgyvb2  (H)
#   7) libiconv/1.17-jjpb4sl         (H)  16) gdbm/1.23                          25) perl/5.40.0                      34) spdlog/1.10.0
#   8) libxml2/2.13.5                     17) gmake/4.4.1                        26) autoconf/2.72
#   9) hwloc/2.11.3-mpich-level-zero      18) cmake/3.30.5                       27) fmt/8.1.1
#
# Notes: with oneapi/release/2025.0.5, you need to set env var UseKmdMigration=1, otherwise PETSc/Kokkos tests will hang.
# Intel is working on a fix.

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=mpicc',
    '--with-cxx=mpicxx',
    '--with-fc=mpifort',
    '--with-debugging=0',
    '--with-mpiexec-tail=gpu_tile_compact.sh',
    '--SYCLPPFLAGS=-Wno-tautological-constant-compare',
    '--with-sycl',
    '--with-syclc=icpx',
    '--with-sycl-arch=pvc',
    '--COPTFLAGS=-O2 -g',
    '--FOPTFLAGS=-O2 -g',
    '--CXXOPTFLAGS=-O2 -g',
    '--SYCLOPTFLAGS=-O2 -g',
    '--download-kokkos',
    '--download-kokkos-kernels',
    '--download-hypre',
  ]
  configure.petsc_configure(configure_options)
