#!/usr/bin/python3

# Use GNU compilers:
#
# Note cray-libsci provides BLAS etc. In summary, we have
# module use /soft/modulefiles
# module unload darshan
# module load PrgEnv-gnu cray-libsci nvhpc-mixed craype-accel-nvidia80 cudatoolkit-standalone/12.4.1
# export MPICH_GPU_SUPPORT_ENABLED=1
# export MPICH_GPU_IPC_ENABLED=0
#
# $ module list
# Currently Loaded Modules:
#   1) libfabric/1.15.2.0       7) nghttp2/1.57.0-ciat5hu         13) cray-mpich/8.1.28   19) cray-libsci/23.12.5
#   2) craype-network-ofi       8) curl/8.4.0-2ztev25             14) cray-pmi/6.1.13     20) nvhpc-mixed/23.9
#   3) perftools-base/23.12.0   9) cmake/3.27.7                   15) cray-pals/1.3.4     21) craype-accel-nvidia80
#   4) darshan/3.4.4           10) cudatoolkit-standalone/12.4.1  16) cray-libpals/1.3.4
#   5) gcc-native/12.3         11) craype/2.7.30                  17) craype-x86-milan
#   6) spack-pe-base/0.6.1     12) cray-dsmml/0.2.2               18) PrgEnv-gnu/8.5.0
#
# With above, Cray-MPICH GPU-aware MPI works on a node but still fail with multiple nodes. In the latter case, you can
# add the PETSc runtime option -use_gpu_aware_mpi 0 as a workaround.

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
    '--download-umpire',
    '--download-hypre',
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
