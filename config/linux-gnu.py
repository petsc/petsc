#!/usr/bin/env python

configure_options = [
  '--with-mpi-include=/home/petsc/soft/linux-rh73/mpich-1.2.4/include',
  '--with-mpi-lib=[/home/petsc/soft/linux-rh73/mpich-1.2.4/lib/libmpich.a,/home/petsc/soft/linux-rh73/mpich-1.2.4/lib/libpmpich.a]',
  '--with-mpirun=mpirun -all-local',
  '--with-gcov=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = [
  '--with-debugging=0',
  '--with-lang=c++']
