#!/usr/bin/python3

# Follow instructions at https://www.alcf.anl.gov/support-center/aurorasunspot/getting-started-aurora
# to set up the proxy settings in your .bashrc and git with SSH protocol in your .ssh/config

# module use /soft/modulefiles
# module load spack-pe-oneapi cmake python
# module load  oneapi/eng-compiler/2023.10.15.002
#
# Currently Loaded Modules:
# 1) gcc/11.2.0            5) spack-pe-gcc/0.5-rc1         9) mpich/52.2-256/icc-all-pmix-gpu
# 2) libfabric/1.15.2.0    6) spack-pe-oneapi/0.5-rc1     10) intel_compute_runtime/release/agama-devel-682.22
# 3) cray-pals/1.2.12      7) cmake/3.26.4-gcc-testing    11) oneapi/eng-compiler/2023.10.15.002
# 4) cray-libpals/1.2.12   8) python/3.10.10-gcc-testing

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
  ]
  configure.petsc_configure(configure_options)
