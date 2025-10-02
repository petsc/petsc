#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-cudac=0',
  '--with-debugging=0',
  #'--with-cc=mpicc.openmpi',
  #'--with-cxx=mpicxx.openmpi',
  #'--with-fc=mpif90.openmpi',
  #'--with-mpiexec=mpiexec.openmpi',
  '--download-openmpi=https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.bz2', # way too many CI failures with 5.0.5
  '--download-mpe=1',
  '--download-fblaslapack=1',
  '--download-hypre=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-suitesparse=1',
  '--download-triangle=1',
  '--download-triangle-build-exec=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-spai=1',
  '--download-parms=1',
  '--download-moab=1',
  '--download-chaco=1',
  '--download-fftw=1',
  '--with-petsc4py=1',
  '--download-mpi4py=1',
  '--download-saws',
  '--download-concurrencykit=1',
  '--download-revolve=1',
  '--download-cams=1',
  '--download-p4est=1',
  '--with-zlib=1',
  '--download-mfem=1',
  '--download-glvis=1',
  '--with-opengl=1',
  '--download-libpng=1',
  '--download-libjpeg=1',
  '--download-slepc=1',
  '--download-slepc-configure-arguments="--with-slepc4py"',
  '--download-hpddm=1',
  '--download-bamg=1',
  '--download-mmg=1',
  '--download-parmmg=1',
  '--download-htool=1',
  '--download-egads=1',
  '--download-opencascade=1',
  '--with-strict-petscerrorcode',
  '--with-coverage',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
