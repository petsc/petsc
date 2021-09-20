#!/usr/bin/python

# Example configure script for the IBM POWER9 and NVIDIA Volta GV100 "Summit" system at OLCF/ORNL.
# This may also be useful for the related Sierra system at LLNL, or other, similar systems that may appear.
# A compiler module and the 'cmake' and 'cuda' modules should be loaded on Summit.
# See inline comments below on other modules that might need to be loaded. 

if __name__ == '__main__':
  import os
  import sys
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    # We use the IBM Spectrum MPI compiler wrappers, regardless of the underlying compilers used.
    '--with-cc=mpicc',
    '--with-cxx=mpiCC',
    '--with-fc=mpifort',

    '--with-make-np=8', # Must limit size of parallel build on Summit login nodes to be within resource quota imposed by OLCF.

    '--with-shared-libraries=1',

    ############################################################
    # Specify compiler optimization flags.
    ############################################################

    # The GCC, PGI, and IBM XL compilers are supported on Summit.
    # Make sure that the correct compiler suite module is loaded,
    #   module load gcc, pgi, or xl
    # and then comment/uncomment the appropriate stanzas below.
    # For optimized cases, more aggressive compilation flags can be tried,
    # but the examples below provide a reasonable start.

    # If a debug build is desired, use the following for any of the compilers:
    #'--with-debugging=yes',
    #'COPTFLAGS=-g',
    #'CXXOPTFLAGS=-g',
    #'FOPTFLAGS=-g',

    # For production builds, disable PETSc debugging support:
    '--with-debugging=no',

    # Optimized flags for PGI:
    'COPTFLAGS=-g -fast',
    'CXXOPTFLAGS=-g -fast',
    'FOPTFLAGS=-g -fast',

    # Optimized flags for XL or GCC:
    #'--COPTFLAGS=-g -Ofast -mcpu=power9',
    #'--CXXOPTFLAGS=-g -Ofast -mcpu=power9',
    #'--FOPTFLAGS=-g -Ofast -mcpu=power9',

    ############################################################
    # Specify BLAS and LAPACK.
    ############################################################

    # Note: ESSL does not provide all functions used by PETSc, so we link netlib LAPACK as well.
    # On ORNL's Summit, one must 'module load' both the essl AND netlib-lapack modules:
    '--with-blaslapack-lib=-L' + os.environ['OLCF_ESSL_ROOT'] + '/lib64 -lessl -llapack -lessl',

    # An alternative in case of difficulty with ESSL is to download/build a portable implementation such as:
    #'--download-fblaslapack=1',
    #'--download-f2cblaslapack', '--download-blis',

    ############################################################
    # Enable GPU support through CUDA/CUSPARSE and ViennaCL.
    ############################################################

    '--with-cuda=1',
    '--with-cudac=nvcc',
    # nvcc reqires the user to specify host compiler name via "-ccbin" when using non-GCC compilers:
    'CUDAFLAGS=-ccbin pgc++',  # For PGI
    #'CUDAFLAGS=-ccbin xlc++_r',  # For IBM XL

    '--download-viennacl=1',

    ############################################################
    # Now specify some commonly used optional packages.
    ############################################################

    '--with-hdf5-dir=' + os.environ['OLCF_HDF5_ROOT'],  # 'module load hdf5' to use the OLCF-provided build
    '--download-metis=1',
    '--download-parmetis=1',
    '--download-triangle=1',
    '--download-ctetgen=1',

    # The options below do not work with the IBM XL compilers.
    # Trying to use the OLCF-provided 'hypre' module also does not work.
    '--download-hypre=1',
    '--download-ml=1',

  ] 
  configure.petsc_configure(configure_options)
