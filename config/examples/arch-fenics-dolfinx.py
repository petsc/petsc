#!/usr/bin/env python3

configure_options = [
  '--download-fenics-dolfinx',
  '--download-metis',
  '--download-parmetis',
  '--download-ptscotch',
  '--download-suitesparse',
  '--download-scalapack',
  '--download-mumps',
  '--download-mpi4py',
  '--download-slepc',
  '--with-petsc4py',
  '--download-boost',
  '--download-fenics-basix',
  '--download-cffi',
  '--download-fenics_ffcx',
  '--download-pathspec',
  '--download-fenics-ufl',
  '--download-scikit_build_core',
  '--download-hdf5',
  '--with-hdf5-cxx-bindings',
  '--download-nanobind',
  '--download-pugixml',
  '--download-spdlog',
  '--download-bison'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
