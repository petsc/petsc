#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=/home/petsc/soft/osx-frisbee/mpich-3.1.3/bin/mpicc',
    '--with-fc=/home/petsc/soft/osx-frisbee/mpich-3.1.3/bin/mpif90',
    '--with-cxx=/home/petsc/soft/osx-frisbee/mpich-3.1.3/bin/mpicxx',
    '--with-clanguage=c++',
    '--with-shared-libraries=yes',
    '--with-debugging=yes',
    '--with-scalar-type=real',
    '--with-64-bit-indices=no',
    '--with-precision=single',
    '--with-sowing=1',
    '--with-sowing-dir=/home/petsc/soft/osx-frisbee/sowing-1.1.17-p1'
    ]
  configure.petsc_configure(configure_options)
