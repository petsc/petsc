#!/usr/bin/env python3

configure_options = [
  '--download-metis',
  '--download-ptscotch',
  '--download-suitesparse',
  '--download-zlib',
  '--download-slepc',
  '--download-hwloc',
  '--download-mumps',
  '--download-scalapack',
  '--download-mpi4py',
  '--download-fftw',
  '--with-petsc4py',
  '--download-boost',
  '--download-pnetcdf',
  '--download-netcdf',
  '--download-bison',
  '--download-hdf5',
  '--download-superlu_dist',
  '--download-hypre',
  '--download-bison',
  '--download-pybind11',
  '--download-rtree',
  '--download-libsupermesh',  
  '--download-firedrake',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
