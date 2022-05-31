#!/usr/bin/python3

#  Modules loaded by default (on login to Crusher):
#
# 1) craype-x86-trento                       9) craype/2.7.15
# 2) libfabric/1.15.0.0                     10) cray-dsmml/0.2.2
# 3) craype-network-ofi                     11) cray-mpich/8.1.16
# 4) perftools-base/22.05.0                 12) cray-libsci/21.08.1.2
# 5) xpmem/2.3.2-2.2_7.8__g93dd7ee.shasta   13) PrgEnv-cray/8.3.3
# 6) cray-pmi/6.1.2                         14) xalt/1.3.0
# 7) cray-pmi-lib/6.0.17                    15) DefApps/default
# 8) cce/14.0.0
#
# Need to load additional rocm module to build with hip
#
# module load rocm/5.1.0
#
# We use Cray Programming Environment, Cray compilers, Cray-mpich.
# To enable GPU-aware MPI, one has to also set this runtime environment variable
#
# export MPICH_GPU_SUPPORT_ENABLED=1
#
# Additional note: "craype-accel-amd-gfx90a" module is recommended for
# "OpenMP offload" or "GPU enabled MPI". It requires "--with-openmp" option.
# [otherwise building c examples gives link errors (when fortran bindings are enabled)]
# Alternative is to use "-lmpi_gtl_hsa" as used below.
#
# ld.lld: error: /autofs/nccs-svm1_home1/balay/petsc/arch-olcf-crusher/lib/libpetsc.so: undefined reference to .omp_offloading.img_start.cray_amdgcn-amd-amdhsa [--no-allow-shlib-undefined]
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
    #'--with-openmp=1', # enable if using "craype-accel-amd-gfx90a" module
    '--with-debugging=0',
    '--with-mpiexec=srun -p batch -N 1 -A csc314_crusher -t 00:10:00',
    '--with-hip',
    '--with-hipc=hipcc',
    '--download-kokkos',
    '--download-kokkos-kernels',
  ]
  configure.petsc_configure(configure_options)
