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
    #'--with-debugging=0', # TODO: fix mat_tests-ex62_14_mpiaijcusparse_cpu mat_tests-ex62_14_mpiaijcusparse_seq_cpu
    '--with-cuda',
    '--with-openmp',
    '--with-shared-libraries',
    '--download-kokkos', # Kokkos-5.0 requires c++20 and cuda-12.2 or above
    '--download-kokkos-kernels',
    '--with-strict-petscerrorcode',
  ]

  configure.petsc_configure(configure_options)
