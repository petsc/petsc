#!/usr/bin/env python

configure_options = [
  '--with-mpi-include=/home/petsc/soft/linux-rh73/mpich-1.2.4/include',
  '--with-mpi-lib=[/home/petsc/soft/linux-rh73/mpich-1.2.4/lib/libmpich.a,/home/petsc/soft/linux-rh73/mpich-1.2.4/lib/libpmpich.a]',
  '--with-mpirun=mpirun -all-local',
  '--with-cc=gcc',
  #blocksolve95
  '--with-blocksolve95-lib=/home/petsc/software/BlockSolve95/lib/libO/linux/libBS95.a',
  #dscpack
  '--with-dscpack-dir=/home/petsc/soft/linux-rh73/DSCPACK1.0',
  #spooles
  '--with-spooles-dir=/home/petsc/soft/linux-rh73/spooles-2.2',
  #superlu, superlu_dist
  '--with-superlu-dir=/home/petsc/soft/linux-rh73/SuperLU_3.0',
  '--with-superlu_dist-lib=/home/petsc/soft/linux-rh73/SuperLU_DIST_2.0-Jul_21_2004/superlu_linux.a',
  #umfpack
  '--with-umfpack-dir=/home/petsc/soft/linux-rh73/UMFPACKv4.3/UMFPACK'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
