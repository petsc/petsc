#!/usr/bin/python

# Example configure script for Cray XC-series systems with Intel "Knights 
# Landing" (KNL) processors.
# This script was constructed for and tested on the Cori XC40 system, but 
# should work (or be easily modified to do so) on other Cray XC-series systems.

if __name__ == '__main__':
  import os
  import sys
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    # We use the Cray compiler wrappers below, regardless of what underlying 
    # compilers we are actually using.
    '--with-cc=cc',
    '--with-cxx=CC',
    '--with-fc=ftn',
    
    # Cray supports the use of Intel, Cray, or GCC compilers.
    # Make sure that the correct programming environment module is loaded, 
    # and then comment/uncomment the apprpriate stanzas below.

    # Flags for the Intel compilers:
    # NOTE: For versions of the Intel compiler < 18.x, one may need to specify
    # the undocumented compiler option '-mP2OPT_hpo_vec_remainder=F', which 
    # disables generation of vectorized remainder loops; this works around 
    # some incorrect code generation. This option should NOT be used with later 
    # compiler versions -- it is detrimental to performance, and the behavior 
    # may change without warning because this is an undocumented option.
    '--COPTFLAGS=-g -xMIC-AVX512 -O3',
    '--CXXOPTFLAGS=-g -xMIC-AVX512 -O3',
    '--FOPTFLAGS=-g -xMIC-AVX512 -O3',
    # Use  BLAS and LAPACK provided by Intel MKL.
    # (Below only works when PrgEnv-intel is loaded; it is possible, but not 
    # straightfoward, to use MKL on Cray systems with non-Intel compilers.)
    # If Cray libsci is preferred, comment out the line below.
    '--with-blaslapack-lib=-mkl -L' + os.environ['MKLROOT'] + '/lib/intel64',
    # Prefer hand-coded kernels using AVX-512 intrinsics when available.
    '--with-avx512-kernels=1',

    # Flags for the Cray compilers:
#    '--COPTFLAGS=-g -hcpu=mic-knl'
#    '--CXXOPTFLAGS=-g -hcpu=mic-knl'
#    '--FOPTFLAGS=-g -hcpu=mic-knl'

    # Flags for the GCC compilers:
#    '--COPTFLAGS=-g -march=knl -O3 -mavx512f -mavx512cd -mavx512er -mavx512pf',
#    '--CXXOPTFLAGS=-g -march=knl -O3 -mavx512f -mavx512cd -mavx512er -mavx512pf',
#    '--FOPTFLAGS=-g -march=knl -O3 -mavx512f -mavx512cd -mavx512er -mavx512pf',

    '--with-debugging=no',
    '--with-memalign=64',
    '--with-mpiexec=srun', # Some systems (e.g., ALCF Theta) use '--with-mpiexec=aprun' instead.
    '--known-mpi-shared-libraries=1',
    '--with-clib-autodetect=0',
    '--with-fortranlib-autodetect=0',
    '--with-cxxlib-autodetect=0',
    '--LIBS=-lstdc++',
    '--LDFLAGS=-dynamic', # Needed if wish to use dynamic shared libraries.

    # Below "--known-" options are from the "reconfigure.py" script generated 
    # after an intial configure.py run using '--with-batch'.
    '--known-level1-dcache-size=32768',
    '--known-level1-dcache-linesize=64',
    '--known-level1-dcache-assoc=8',
    '--known-sdot-returns-double=0',
    '--known-snrm2-returns-double=0',
    '--known-has-attribute-aligned=1',
    '--known-64-bit-blas-indices=0',
    '--with-batch=1',
  ]
  configure.petsc_configure(configure_options)
