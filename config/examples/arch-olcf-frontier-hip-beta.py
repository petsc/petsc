#!/usr/bin/python

# Example configure script for Cray systems with AMD Epyc CPU's  and AMD GPGPU's
# as for example Frontier.  Since Frontier does not exist yet, this is a beta
# file
#
# Recommended module loads 
#    module load PrgEnv-cray
#    module load cmake
#    module load rocm-alt/3.5.0
#    module load blas
#    module load lapack

# Note:  For GPU-aware MPI, you need something like this:
#module use /home/users/twhite/share/modulefiles
#module load ompi
#export GCC_X86_64="$GCC_PATH/snos"
#hipcc -g -Ofast --amdgpu-target=gfx906,gfx908 -I$MPI_HOME/include -gcc-toolchain $GCC_X86_64 -c file.cpp
#hipcc -g -Ofast --amdgpu-target=gfx906,gfx908 -gcc-toolchain $GCC_X86_64 file.o -L$MPI_HOME/lib -lmpi
#srun -N1 -n4 -p amdMI60 --exclusive --time=5:00 ./a.out

import os

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--prefix='+os.path.join(os.environ['HOME'],'software','petsc'),
    '--with-hip=1',
    '--with-hipcc=hipcc',
    '--with-precision=double',
    '--with-clanguage=c',
    '--with-cc=mpicc',
    '--with-cxx=mpicxx',
    '--with-fc=mpifort',
    '--with-mpiexec=srun', 
    '--with-blaslapack-lib=-L'+os.environ['BLASDIR']+' -L'+os.environ['LAPACK_DIR']+' -llapack -lblas',
    '--with-shared-libraries=0', 

    # The GCC and Clang compilers are supported on tulip/frontier
    # Load modules as described above
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

    # Optimized flags:
    'COPTFLAGS=-g -fast',
    'CXXOPTFLAGS=-g -fast',
    'HIPOPTFLAGS=-Ofast --amdgpu-target=gfx906,gfx908',
    'FOPTFLAGS=-g -fast',

  ]
  configure.petsc_configure(configure_options)
