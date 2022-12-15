#!/usr/bin/python3

# Due to a series of issues (nvhpc versions etc), we don't have a workaround
# to use the Polaris default modules "PrgEnv-cray nvhpc" to build petsc
# with Kokkos, we use PrgEnv-gnu instead. One needs to
#
# module load cudatoolkit-standalone PrgEnv-gnu cray-libsci
#
# Note cray-libsci provides BLAS etc. In summary, we have
#
# $ module list

# Currently Loaded Modules:
#   1) craype-x86-rome          5) cmake/3.23.2                    9) gcc/11.2.0         13) cray-pmi/6.1.2       17) PrgEnv-gnu/8.3.3
#   2) libfabric/1.11.0.4.125   6) craype-accel-nvidia80          10) craype/2.7.15      14) cray-pmi-lib/6.0.17
#   3) craype-network-ofi       7) cray-libsci/21.08.1.2          11) cray-dsmml/0.2.2   15) cray-pals/1.1.7
#   4) perftools-base/22.05.0   8) cudatoolkit-standalone/11.8.0  12) cray-mpich/8.1.16  16) cray-libpals/1.1.7

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=cc',
    '--with-cxx=CC',
    '--with-fc=ftn',
    '--with-debugging=0',
    '--with-cuda',
    '--with-cudac=nvcc',
    '--with-cuda-arch=80', # Since there is no easy way to auto-detect the cuda arch on the gpu-less Polaris login nodes, we explicitly set it.
    '--download-kokkos',
    '--download-kokkos-kernels',
  ]
  configure.petsc_configure(configure_options)

