#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-cc=mpicc -cc=icx', # need to make mpicc/mpicxx also SYCL compilers.
    '--with-cxx=mpicxx -cxx=dpcpp', # Intel MPI does not accept -cxx=icpx, though it should.
    '--with-fc=0',
    '--COPTFLAGS=-g -O2',
    '--CXXOPTFLAGS=-g -O2',
    '--CXXPPFLAGS=-std=c++17',
    '--SYCLOPTFLAGS=-g -O2',
    # To supress warnings in checking Kokkos-Kernels headers like:
    # Kokkos_MathematicalFunctions.hpp:299:34: warning: comparison with infinity always evaluates
    # to false in fast floating point modes [-Wtautological-constant-compare]
    # KOKKOS_IMPL_MATH_UNARY_PREDICATE(isinf)
    '--SYCLPPFLAGS=-Wno-tautological-constant-compare',
    '--download-kokkos=1',
    '--downoad-kokkos-kernels=1',
    '--with-cuda=0',
    '--with-sycl=1',
    '--with-syclc=dpcpp',
    '--with-sycl-dialect=c++17',
  ]

  configure.petsc_configure(configure_options)
