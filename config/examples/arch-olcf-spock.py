#!/usr/bin/python3

#  Modules loaded by default (on login to spock):
#
#  1) craype-x86-rome                          8) cce/12.0.3
#  2) libfabric/1.11.0.4.75                    9) craype/2.7.11
#  3) craype-network-ofi                      10) cray-dsmml/0.2.2
#  4) perftools-base/21.10.0                  11) cray-mpich/8.1.10
#  5) xpmem/2.2.40-2.1_2.44__g3cf3325.shasta  12) cray-libsci/21.08.1.2
#  6) cray-pmi/6.0.14                         13) PrgEnv-cray/8.2.0
#  7) cray-pmi-lib/6.0.14                     14) DefApps/default
#
# Need to load additional rocm module to build with hip
#
# module load rocm/4.3.0
#
# Note: LIBS option below is needed to use GPU enabled MPI. It also requires
# the following env variables to be set at runtime
#
# export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
# export MPICH_GPU_SUPPORT_ENABLED=1
# export MPICH_SMP_SINGLE_COPY_MODE=CMA
#
# Additional note: If "craype-accel-amd-gfx908" module is loaded (that is
# needed for "OpenMP offload") - it causes link errors when using 'cc or hipcc'
# with fortran objs, hence not used
#

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=cc',
    '--with-cxx=CC',
    '--with-fc=ftn',
    'LIBS=-L{x}/gtl/lib -lmpi_gtl_hsa'.format(x=os.environ['CRAY_MPICH_ROOTDIR']),
    '--with-debugging=0',
    '--with-mpiexec=srun -p ecp -N 1 -A csc314 -t 00:10:00',
    '--with-hip=1',
    '--with-hipc=hipcc',
    '--download-kokkos=1',
    '--download-kokkos-kernels=1',
    '--download-magma=1',
    '--with-magma-gputarget=gfx908',
  ]
  configure.petsc_configure(configure_options)
