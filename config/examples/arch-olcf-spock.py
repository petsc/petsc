#!/usr/bin/python3

#As suggested from OLCF staff this is my rc file
#
#module load craype-accel-amd-gfx908
#module load PrgEnv-cray
#module load rocm
#export PE_MPICH_GTL_DIR_amd_gfx908="-L/opt/cray/pe/mpich/8.1.4/gtl/lib"
#export PE_MPICH_GTL_LIBS_amd_gfx908="-lmpi_gtl_hsa"
#export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
#export MPICH_GPU_SUPPORT_ENABLED=1
#export MPICH_SMP_SINGLE_COPY_MODE=CMA
#

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    # When we compile HIP code in PETSc, we eventually include mpi.h.
    # MPI include folder is hidden by cc/CC and PETSc does not detect it
    '--HIPPPFLAGS=-I'+os.environ['MPICH_DIR']+'include',
    # Needed by MPICH:
    # ld.lld: error: /opt/cray/pe/mpich/8.1.4/gtl/lib/libmpi_gtl_hsa.so: undefined reference to hsa_amd_memory_pool_allocate
    # and many others
    '--LDFLAGS=-L'+os.environ['ROCM_PATH']+'lib -lhsa-runtime64',
    '--PETSC_ARCH=arch-spock-debug',
    '--download-magma=1',
    '--with-64-bit-indices=0',
    '--with-cc=cc',
    '--with-cxx=CC',
    '--with-debugging=1',
    '--with-fc=ftn',
    '--with-fortran-bindings=0',
    '--with-hip=1',
    '--with-hipc=hipcc',
    '--with-magma-fortran-bindings=0',
    '--with-magma-gputarget=gfx908',
    '--with-mpiexec=srun -p ecp -N 1 -A csc314 -t 00:10:00',
    '--with-precision=double',
    '--with-scalar-type=real',
  ]
  configure.petsc_configure(configure_options)
