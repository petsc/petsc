#!/home/petsc/soft/linux-Ubuntu_12.04-x86_64/Python-2.4.6/bin/python2.4
# Test python2.4 && cmake

configure_options = [
  '--with-debugging=0',
  #'--with-cc=mpicc.openmpi',
  #'--with-cxx=mpicxx.openmpi',
  #'--with-fc=mpif90.openmpi',
  #'--with-mpiexec=mpiexec.openmpi',
  '--download-openmpi=1',
  '--download-fblaslapack=1',
  '--download-hypre=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-suitesparse=1',
  '--download-triangle=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-elemental=1',
  '--with-cxx-dialect=C++11',
  '--download-spai=1',
  '--download-parms=1',
  '--download-moab=1',
  '--download-chaco=1',
  '--download-fftw=1',
  '--download-petsc4py=1',
  '--download-mpi4py=1',
  '--download-saws',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
