#!/usr/bin/env python

configure_options = [
 '--with-mpi-dir=/soft/apps/packages/mpich-gm-1.2.6..13b-intel-8.1-1',
 '--with-blas-lapack-dir=/soft/com/packages/mkl_7.2/mkl72/lib/32',
 '--with-hypre-dir=/soft/apps/packages/petsc-packages/linux-rhAS3-intel81/hypre-1.8b2-mod',
 '--with-spooles-dir=/soft/apps/packages/petsc-packages/linux-rhAS3-intel81/spooles-2.2',
 '--with-superlu-include=/soft/apps/packages/petsc-packages/linux-rhAS3-intel81/SuperLU_3.0-Oct_23_2003/SRC',
 '--with-superlu-lib=/soft/apps/packages/petsc-packages/linux-rhAS3-intel81/SuperLU_3.0-Oct_23_2003/superlu_linux-rhAS3-intel81.a',
 '--with-superlu_dist-include=/soft/apps/packages/petsc-packages/linux-rhAS3-intel81/SuperLU_DIST_2.0-Jul_21_2004/SRC',
 '--with-superlu_dist-lib=/soft/apps/packages/petsc-packages/linux-rhAS3-intel81/SuperLU_DIST_2.0-Jul_21_2004/superlu_linux-rhAS3-intel81.a',
 '-COPTFLAGS=-O2 -tpp6 -xW',
 '-CXXOPTFLAGS=-O2 -tpp6 -xW',
 '-FOPTFLAGS=-O2 -tpp6 -xW',
 '--with-debugging=0'
 ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
