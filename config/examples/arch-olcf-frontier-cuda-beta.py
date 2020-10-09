#!/usr/bin/python

import os

# Example configure script for Cray systems with AMD Epyc CPU's  and AMD GPGPU's
# as for example Frontier, or the Frontier COE development machine tulip
# Recommended module loads 
#    module load cray-mvapich2_gnu
#    module load cmake
#    module load cuda10.2
#    module load blas
#    module load lapack

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--prefix='+os.path.join(os.environ['HOME'],'software','petsc-cuda'),
    '--with-make-test-np=2',
    '--with-precision=double',
    '--with-clanguage=c',
    '--with-cc=mpicc',
    '--with-cxx=mpicxx',
    '--with-fc=mpifort',
    '--with-mpiexec=srun', 
    '--with-blaslapack-lib=-L'+os.environ['BLASDIR']+' -L'+os.environ['LAPACK_DIR']+' -llapack -lblas',
    '--with-shared-libraries=0', 

    '--with-cuda=1',
    '--with-cudac=nvcc',
    '--with-cuda-gencodearch=70',
  ]
  configure.petsc_configure(configure_options)
