#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=/usr/bin/mpicc',
    '--with-fc=/usr/bin/mpif90',
    '--with-cxx=/usr/bin/mpicxx',
    '--with-clanguage=c',
    '--with-shared-libraries=no',
    '--with-debugging=yes',
    '--with-scalar-type=real',
    '--with-64-bit-indices=no',
    '--with-precision=__float128',
    '--with-blas-lapack-lib=[/home/sarich/software/quadf2cblaslapack/libf2clapack.a,/home/sarich/software/quadf2cblaslapack/libf2cblas.a]',
    '--with-sowing=1',
    '--with-sowing-dir=/home/petsc/soft/linux-Ubuntu_12.04-x86_64/sowing-1.1.17-p1'
    ]
  configure.petsc_configure(configure_options)
