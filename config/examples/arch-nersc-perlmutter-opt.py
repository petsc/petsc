#!/usr/bin/env python3

# Example configure script for Perlmutter, the HPE Cray EX system at NERSC/LBNL equipped with
# wth AMD EPYC CPUS and NVIDIA A100 GPUS. Here we target the GPU compute nodes and builds with
# support for the CUDA/cuSPARSE, Kokkos, and ViennaCL back-ends. 
#
# Currently, configuring PETSc on the system does not require loading many , if any, non-default modules.
# As documented at https://docs.nersc.gov/systems/perlmutter/software/#mpi, typical settings might be
#
#   export MPICH_GPU_SUPPORT_ENABLED=1
#   module load cudatoolkit
#   module load PrgEnv-gnu
#   module load craype-accel-nvidia80
#
# The above are currently present in the default environment. Users may wish to 'module load' a
# different programming environment (which will generally force a reload of certain related modules,
# such as the one corresponding to the MPI implementation).

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-make-np=8', # Must limit size of parallel build to stay within resource limitations imposed by the center
    '--with-mpiexec=srun -G4', # '-G4' requests all four GPUs present on a Perlmutter GPU compute node.
    '--with-batch=0',

    # Use the Cray compiler wrappers, regardless of the underlying compilers loaded by the programming environment module:
    '--with-cc=cc',
    '--with-cxx=CC',
    '--with-fc=ftn',

    # Build with aggressive optimization ('-O3') but also include debugging symbols ('-g') to support detailed profiling.
    # If you are doing development, using no optimization ('-O0') can be a good idea. Also note that some compilers (GNU
    # is one) support the '-g3' debug flag, which allows macro expansion in some debuggers; this can be very useful when
    # debugging PETSc code, as PETSc makes extensive use of macros.
    '--COPTFLAGS=   -g -O3',
    '--CXXOPTFLAGS= -g -O3',
    '--FOPTFLAGS=   -g -O3',
    '--CUDAFLAGS=   -g -O3',
    '--with-debugging=0',  # Disable debugging for production builds; use '--with-debugging=1' for development work.

    # Set sowing-cc and sowing-cxx explicitly, as this prevents errors caused by compiling sowing with GCC when a
    # programming environment other than PrgEnv-gnu has been loaded. If there is this compiler mismatch, we will see
    # errors like
    # 
    #   /opt/nvidia/hpc_sdk/Linux_x86_64/22.5/compilers/include/bits/floatn.h:60:17: error: two or more data types in declaration specifiers
    #   typedef float _Float32;
    #                 ^~~~~~~~
    '--download-sowing-cc=cc', # Note that sowing is only needed when Fortran bindings are required.
    '--download-sowing-cc=CC',


    # Build with support for CUDA/cuSPARSE, Kokkos/Kokkos Kernels, and ViennaCL back-ends:
    '--with-cuda=1',
    '--with-cuda-arch=80',
    '--download-viennacl',
    '--download-kokkos',
    '--download-kokkos-kernels',
    '--with-kokkos-kernels-tpl=0', # Use native Kokkos kernels, rather than NVIDIA-provided ones.

    # Download and build a few commonly-used packages:
    '--download-hypre',
    '--download-metis',
    '--download-parmetis',
    '--download-hdf5', # Note that NERSC does provide an HDF5 module, but using our own is generally reliable.
    '--download-hdf5-fortran-bindings',
  ]
  configure.petsc_configure(configure_options)
