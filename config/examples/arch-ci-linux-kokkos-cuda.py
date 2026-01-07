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
    '--with-make-test-np=3',
    '--with-debugging=0',
    '--with-cuda',
    '--with-openmp',
    '--with-shared-libraries',
    '--download-kokkos',
    '--download-kokkos-kernels',
    '--with-strict-petscerrorcode',
  ]

  configure.petsc_configure(configure_options)
