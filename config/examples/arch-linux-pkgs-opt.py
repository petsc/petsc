#!/home/petsc/soft/linux-Ubuntu_12.04-x86_64/Python-2.4.6/bin/python2.4
# Test python2.4 && cmake

configure_options = [
  '--with-debugging=0',
  '--with-cc=mpicc.openmpi',
  '--with-cxx=mpicxx.openmpi',
  '--with-fc=mpif90.openmpi',
  '--with-mpiexec=mpiexec.openmpi',
  '--download-f-blas-lapack=1',
  '--download-hypre=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-umfpack=1',
  '--download-triangle=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-elemental=1',
  '--with-cxx-dialect=C++11',
  '--download-spai=1',
  '--download-chaco=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
