#!/usr/bin/env python

configure_options = [
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  '--with-scalar-type=complex',
  'FFLAGS=-ftrap=%none',
  'DATAFILESPATH=/export/home/petsc/datafiles',
  '--with-c2html=0',
  '--with-sowing-dir=/export/home/glci/soft/sowing-1.1.25-p1',
  '--with-mpi-dir=/export/home/glci/soft/cmplx-pkgs-dbg',
  '--with-metis-dir=/export/home/glci/soft/cmplx-pkgs-dbg',
  '--with-parmetis-dir=/export/home/glci/soft/cmplx-pkgs-dbg',
  '--with-triangle-dir=/export/home/glci/soft/cmplx-pkgs-dbg',
  '--with-superlu-dir=/export/home/glci/soft/cmplx-pkgs-dbg',
  '--with-blaslapack-dir=/export/home/glci/soft/cmplx-pkgs-dbg/lib',
  '--with-scalapack-dir=/export/home/glci/soft/cmplx-pkgs-dbg',
  '--with-mumps-dir=/export/home/glci/soft/cmplx-pkgs-dbg',
  '--with-hdf5-dir=/export/home/glci/soft/cmplx-pkgs-dbg',
  '--with-suitesparse-dir=/export/home/glci/soft/cmplx-pkgs-dbg',
  '--with-chaco-dir=/export/home/glci/soft/cmplx-pkgs-dbg',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
