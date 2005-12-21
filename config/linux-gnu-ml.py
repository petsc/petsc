#!/usr/bin/env python

# ******* Currently not tested **********

configure_options = [
  '--with-mpi-include=/home/petsc/soft/linux-rh73/mpich-1.2.4/include',
  '--with-mpi-lib=[/home/petsc/soft/linux-rh73/mpich-1.2.4/lib/libpmpich.a,libmpich.a,libpmpich.a]',
  '--with-mpirun=mpirun -all-local',
  #'--with-cc=gcc',
  '--with-ml-dir=/home/petsc/soft/linux-rh73/trilinos-4.0-ml'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
