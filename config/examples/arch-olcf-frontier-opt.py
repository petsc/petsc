#!/usr/bin/python3

#  Use GNU compilers
#    module load PrgEnv-gnu
#    module load cray-mpich
#    module load craype-accel-amd-gfx90a
#    module load rocm
#
# To enable GPU-aware MPI, one has to also set this runtime environment variable
#
#    export MPICH_GPU_SUPPORT_ENABLED=1
#
# To use hipcc with GPU-aware Cray MPICH, use the following environment variables to setup the needed header files and libraries.
#
#    -I${MPICH_DIR}/include
#    -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}
#
# See also https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#gpu-aware-mpi
#
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-debugging=0',
    '--with-cc=cc',
    '--with-cxx=CC',
    '--with-fc=ftn',
    '--with-mpiexec=srun -p batch -N 1 -A csc314 -t 00:10:00',
    '--with-batch',
    '--with-hip',
    '--with-hipc=hipcc',
    'LIBS={GTLDIR} {GTLLIBS}'.format(GTLDIR=os.environ['PE_MPICH_GTL_DIR_amd_gfx90a'], GTLLIBS=os.environ['PE_MPICH_GTL_LIBS_amd_gfx90a']),
    '--download-metis',
    '--download-parmetis',
    '--download-kokkos',
    '--download-kokkos-kernels',
    '--download-superlu_dist',
    '--download-umpire',
    '--download-hypre'
  ]
  configure.petsc_configure(configure_options)

#  Use Cray compilers
#    module load PrgEnv-cray
#    module load cray-mpich
#    module load amd-mixed/5.4.0

# To enable GPU-aware MPI, one has to also set this runtime environment variable
#
#    export MPICH_GPU_SUPPORT_ENABLED=1
#
# Additional note: "craype-accel-amd-gfx90a" module is recommended for
# "OpenMP offload" or "GPU enabled MPI". It requires "--with-openmp" option.
# [otherwise building c examples gives link errors (when fortran bindings are enabled)]
# Alternative is to use "-lmpi_gtl_hsa" as shown below.
#
#   ld.lld: error: lib/libpetsc.so: undefined reference to .omp_offloading.img_start.cray_amdgcn-amd-amdhsa [--no-allow-shlib-undefined]
#
#  Also, please ignore warnings like this. If you don't use Fortran, use '--with-fc=0' to get rid of them.
#
# ftn-878 ftn: WARNING PETSC, File = ../../../autofs/nccs-svm1_home1/jczhang/petsc/src/tao/ftn-mod/petsctaomod.F90, Line = 37, Column = 13
#  A module named "PETSCVECDEFDUMMY" has already been directly or indirectly use associated into this scope.


# if __name__ == '__main__':
#   import sys
#   import os
#   sys.path.insert(0, os.path.abspath('config'))
#   import configure
#   configure_options = [
#     '--with-debugging=0',
#     '--with-cc=cc',
#     '--with-cxx=CC',
#     '--with-fc=ftn',
#     # -std=c2x is a workaround for this hipsparse problem
#     #   /opt/rocm-5.4.0/include/hipsparse/hipsparse.h:8741:28: error: expected '= constant-expression' or end of enumerator definition
#     #      HIPSPARSE_ORDER_COLUMN [[deprecated("Please use HIPSPARSE_ORDER_COL instead")]] = 1,
#     # -Wno-constant-logical-operand is a workaround to suppress excessive warnings caused by -std=c2x in PETSc source which we don't want to address, see MR !6287
#     '--CFLAGS=-std=c2x -Wno-constant-logical-operand',
#     'LIBS={GTLDIR} {GTLLIBS}'.format(GTLDIR=os.environ['PE_MPICH_GTL_DIR_amd_gfx90a'], GTLLIBS=os.environ['PE_MPICH_GTL_LIBS_amd_gfx90a']),
#     #'--with-openmp=1', # enable if using "craype-accel-amd-gfx90a" module
#     '--with-mpiexec=srun -p batch -N 1 -A csc314 -t 00:10:00',
#     '--with-batch',
#     '--with-hip',
#     '--with-hipc=hipcc',
#     '--download-kokkos',
#     '--download-kokkos-kernels',
#   ]
#   configure.petsc_configure(configure_options)
