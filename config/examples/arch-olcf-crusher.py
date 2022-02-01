#!/usr/bin/python3

#  Modules loaded by default (on login to Crusher):
#
# 1) craype-x86-trento                       9) cce/13.0.0
# 2) libfabric/1.13.0.0                     10) craype/2.7.13
# 3) craype-network-ofi                     11) cray-dsmml/0.2.2
# 4) perftools-base/21.12.0                 12) cray-libsci/21.08.1.2
# 5) xpmem/2.3.2-2.2_1.16__g9ea452c.shasta  13) PrgEnv-cray/8.2.0
# 6) cray-pmi/6.0.16                        14) DefApps/default
# 7) cray-pmi-lib/6.0.16                    15) rocm/4.5.0
# 8) tmux/3.2a                              16) cray-mpich/8.1.12
#
# We use Cray Programming Environment, Cray compilers, Cray-mpich.
# To enable GPU-aware MPI, one has to also set this runtime environment variable
#
# export MPICH_GPU_SUPPORT_ENABLED=1
#
# Additional note: If "craype-accel-amd-gfx90a" module is loaded (that is
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
    '--with-mpiexec=srun -p batch -N 1 -A csc314_crusher -t 00:10:00',
    '--with-hip',
    '--with-hipc=hipcc',
    '--download-kokkos',
    '--download-kokkos-kernels',
  ]
  configure.petsc_configure(configure_options)
