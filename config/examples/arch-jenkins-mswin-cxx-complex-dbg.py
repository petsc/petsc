#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-debugging=0',
    '--with-blaslapack-dir=/home/petsc/soft-win/f2cblaslapack-3.4.2.q3/lib',
    '--with-cc=win32fe cl',
    '--with-cxx=win32fe cl',
    '--with-clanguage=cxx',
    '--with-scalar-type=complex',
    '--with-fc=0',
    '--with-mpi-include=[/cygdrive/c/PROGRA~2/MICROS~2/MPI/Include/,/cygdrive/c/PROGRA~2/MICROS~2/MPI/Include/x64]',
    '--with-mpi-lib=[/cygdrive/c/PROGRA~2/MICROS~2/MPI/lib/x64/msmpifec.lib,/cygdrive/c/PROGRA~2/MICROS~2/MPI/lib/x64/msmpi.lib]',
    '--with-mpiexec=/cygdrive/c/PROGRA~1/MICROS~2/Bin/mpiexec',
    '--with-shared-libraries=0',
    'DATAFILESPATH=c:/cygwin64/home/petsc/datafiles',
  ]
  configure.petsc_configure(configure_options)
