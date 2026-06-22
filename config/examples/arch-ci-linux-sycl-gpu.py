#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-debugging=0',
    '--with-make-np=8',
    '--with-make-test-np=1',
    '--with-cc=mpiicx',
    '--with-cxx=mpiicpx',
    '--with-fc=0',
    '--SYCLPPFLAGS=-Wno-tautological-constant-compare',
    '--with-sycl',
    '--with-syclc=icpx',
    '--with-sycl-arch=pvc',
    '--download-kokkos',
    '--download-kokkos-kernels',
  ]
  configure.petsc_configure(configure_options)
