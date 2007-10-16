#!/bin/sh
set -x

# basic gcc tests
./config/asterix.py
make PETSC_ARCH=asterix all alltests DATAFILESPATH=/home/balay/datafiles 
make PETSC_ARCH=asterix tree DATAFILESPATH=/home/balay/datafiles ACTION=testexamples_C_NoComplex

./config/configure.py  CC=gcc FC=gfortran CXX=g++ PETSC_ARCH=asterix-cxx-sieve --with-clanguage=cxx \
--with-sieve=1 --download-mpich=1 --download-boost=1 --download-chaco=1 --download=jostle=1 \
--download-plapack=1 --download-tetgen=1 --download-triangle=1 --download-hdf5=1
make PETSC_ARCH=asterix-cxx-sieve all testexamples testfortran

./config/asterix-openmpi.py
make PETSC_ARCH=asterix-openmpi all alltests DATAFILESPATH=/home/balay/datafiles 

./config/asterix.py --with-clanguage=cxx -PETSC_ARCH=asterix-cxx-opt --with-debugging=0
make PETSC_ARCH=asterix-cxx-opt all test
./config/asterix.py --with-scalar-type=complex --with-clanguage=cxx -PETSC_ARCH=asterix-complex
make PETSC_ARCH=asterix-complex all test
make PETSC_ARCH=asterix-complex testexamples testfortran 
make PETSC_ARCH=asterix-complex tree DATAFILESPATH=/home/balay/datafiles ACTION=testexamples_C_X11
make PETSC_ARCH=asterix-complex tree DATAFILESPATH=/home/balay/datafiles ACTION=testexamples_C_Complex
make PETSC_ARCH=asterix-complex tree DATAFILESPATH=/home/balay/datafiles ACTION=testexamples_Fortran_Complex

./config/asterix.py --download-prometheus=1 --download-parmetis=1 -PETSC_ARCH=asterix-prometheus \
--with-dscpack=1 --with-dscpack-include=/home/balay/soft/linux-fc/DSCPACK1.0/DSC_LIB \
--with-dscpack-lib=/home/balay/soft/linux-fc/DSCPACK1.0/DSC_LIB/dsclibdbl.a \
--download-umfpack=1
make PETSC_ARCH=asterix-prometheus all test
make PETSC_ARCH=asterix-prometheus tree DATAFILESPATH=/home/balay/datafiles ACTION=testexamples_PARMETIS

# basic intel tests
./config/asterix-intel.py --with-debugging=0 -PETSC_ARCH=asterix-intel-opt --download-spooles=1 --download-superlu=1 \
--download-superlu_dist=1 --download-hypre=1 --download-spai=1 --download-blacs=1 --download-scalapack=1 \
--download-mumps=1 --download-mpe=1 --download-sundials=1 LIBS=/usr/lib/libm.a
make PETSC_ARCH=asterix-intel-opt all test
make PETSC_ARCH=asterix-intel-opt testexamples testfortran
make PETSC_ARCH=asterix-intel-opt tree DATAFILESPATH=/home/balay/datafiles ACTION=testexamples_F90
make PETSC_ARCH=asterix-intel-opt tree DATAFILESPATH=/home/balay/datafiles ACTION=testexamples_SUPERLU
make PETSC_ARCH=asterix-intel-opt tree DATAFILESPATH=/home/balay/datafiles ACTION=testexamples_SUPERLU_DIST
make PETSC_ARCH=asterix-intel-opt tree DATAFILESPATH=/home/balay/datafiles ACTION=testexamples_HYPRE
make PETSC_ARCH=asterix-intel-opt tree DATAFILESPATH=/home/balay/datafiles ACTION=testexamples_SPAI
make PETSC_ARCH=asterix-intel-opt tree DATAFILESPATH=/home/balay/datafiles ACTION=testexamples_MUMPS

./config/asterix-intel.py --with-clanguage=cxx -PETSC_ARCH=asterix-intel-cxx
make PETSC_ARCH=asterix-intel-cxx all test
./config/asterix-intel.py --with-scalar-type=complex --with-clanguage=cxx -PETSC_ARCH=asterix-intel-complex
make PETSC_ARCH=asterix-intel-complex all test

./config/asterix-intel.py --download-prometheus=1 --download-parmetis=1 --with-clanguage=cxx -PETSC_ARCH=asterix-intel-cxx-prometheus
make PETSC_ARCH=asterix-intel-cxx-prometheus all test

# tops test
#./config/asterix-tops.py
#make PETSC_ARCH=asterix-tops all test

./config/asterix-c89.py
make PETSC_ARCH=asterix-c89 all test alltests DATAFILESPATH=/home/balay/datafiles

# basic sun tests
./config/asterix-sun.py
make PETSC_ARCH=asterix-sun all test
./config/asterix-sun.py --with-shared=1 --with-dynamic=1 -PETSC_ARCH=asterix-sun-dynamic
make PETSC_ARCH=asterix-sun-dynamic all alltests DATAFILESPATH=/home/balay/datafiles
./config/asterix-sun.py --with-clanguage=cxx -PETSC_ARCH=asterix-sun-cxx  --with-debugging=0 --with-pic=0 \
--download-f-blaslapack=1 --download-spooles=1 --download-superlu=1 \
--download-superlu_dist=1 --download-hypre=1 --download-spai=1 --download-blacs=1 --download-scalapack=1 \
--download-mumps=1 --download-mpe=1 --download-sundials=1
make PETSC_ARCH=asterix-sun-cxx all
make PETSC_ARCH=asterix-sun-cxx CLINKER=sun-cc shared
make PETSC_ARCH=asterix-sun-cxx test
./config/asterix-sun.py --with-shared=1 --with-dynamic=1 --with-scalar-type=complex --with-clanguage=cxx -PETSC_ARCH=asterix-sun-complex-dynamic
make PETSC_ARCH=asterix-sun-complex-dynamic all
make PETSC_ARCH=asterix-sun-complex-dynamic CLINKER=sun-cc shared
make PETSC_ARCH=asterix-sun-complex-dynamic test

# basic gcc3 tests
./config/asterix-gcc3.py
make PETSC_ARCH=asterix-gcc3 all test
#make PETSC_ARCH=asterix-gcc3-cxx all test
./config/asterix-gcc3.py --with-scalar-type=complex --with-clanguage=cxx -PETSC_ARCH=asterix-gcc3-complex-opt --with-debugging=0
make PETSC_ARCH=asterix-gcc3-complex-opt all test

