#!/usr/bin/python3

# Use GNU compilers:
#
# module load cudatoolkit-standalone PrgEnv-gnu cray-libsci
#
# Note cray-libsci provides BLAS etc. In summary, we have
#
# module load cudatoolkit-standalone/11.8.0 PrgEnv-gnu gcc/10.3.0 cray-libsci
#
# $ module list
# Currently Loaded Modules:
#   1) craype-x86-rome          5) craype-accel-nvidia80           9) cray-dsmml/0.2.2     13) PrgEnv-gnu/8.3.3
#   2) libfabric/1.15.2.0       6) cmake/3.23.2                   10) cray-pmi/6.1.10      14) cray-libsci/23.02.1.1
#   3) craype-network-ofi       7) cudatoolkit-standalone/11.8.0  11) cray-pals/1.2.11     15) gcc/10.3.0
#   4) perftools-base/23.03.0   8) craype/2.7.20                  12) cray-libpals/1.2.11  16) cray-mpich/8.1.25

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

# Use NVHPC compilers
#
# Unset so that cray won't add -gpu to nvc even when craype-accel-nvidia80 is loaded
# unset CRAY_ACCEL_TARGET
# module load nvhpc/22.11 PrgEnv-nvhpc
#
# I met two problems with nvhpc and Kokkos (and Kokkos-Kernels) 4.2.0.
# 1) Kokkos-Kernles failed at configuration to find TPL cublas and cusparse from NVHPC.
#    As a workaround, I just load cudatoolkit-standalone/11.8.0 to let KK use cublas and cusparse from cudatoolkit-standalone.
# 2) KK failed at compilation
# "/home/jczhang/petsc/arch-kokkos-dbg/externalpackages/git.kokkos-kernels/batched/dense/impl/KokkosBatched_Gemm_Serial_Internal.hpp", line 94: error: expression must have a constant value
#     constexpr int nbAlgo = Algo::Gemm::Blocked::mb();
#                            ^
# "/home/jczhang/petsc/arch-kokkos-dbg/externalpackages/git.kokkos-kernels/blas/impl/KokkosBlas_util.hpp", line 58: note: cannot call non-constexpr function "__builtin_is_device_code" (declared implicitly)
#           KOKKOS_IF_ON_HOST((return 4;))
#           ^
#           detected during:
#
# It is a KK problem and I have to wait for their fix.
