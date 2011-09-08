#!/bin/sh
set -x

# basic gcc tests
./config/examples/asterix/asterix64.py
make PETSC_ARCH=asterix64 all alltests tests_DATAFILESPATH DATAFILESPATH=/home/balay/datafiles
./config/configure.py  CC=gcc FC=gfortran CXX=g++ PETSC_ARCH=asterix64-cxx-sieve --with-clanguage=cxx \
--with-sieve=1 --download-mpich=1 --download-boost=1 --download-chaco=1 \
--download-plapack=1 --download-tetgen=1 --download-triangle=1 --download-hdf5=1
make PETSC_ARCH=asterix64-cxx-sieve all alltests tests_DATAFILESPATH DATAFILESPATH=/home/balay/datafiles
./config/examples/asterix/asterix64-openmpi.py
make PETSC_ARCH=asterix64-openmpi all alltests tests_DATAFILESPATH DATAFILESPATH=/home/balay/datafiles
./config/examples/asterix/asterix64.py --with-clanguage=cxx -PETSC_ARCH=asterix64-cxx-opt --with-debugging=0 --with-log=0
make PETSC_ARCH=asterix64-cxx-opt all alltests tests_DATAFILESPATH DATAFILESPATH=/home/balay/datafiles
./config/examples/asterix/asterix64.py --with-scalar-type=complex --with-clanguage=cxx -PETSC_ARCH=asterix64-complex
make PETSC_ARCH=asterix64-complex all test alltests tests_DATAFILESPATH DATAFILESPATH=/home/balay/datafiles

./config/examples/asterix/asterix64.py --download-prometheus=1 --download-parmetis=1 -PETSC_ARCH=asterix64-prometheus \
--download-umfpack=1
make PETSC_ARCH=asterix64-prometheus all alltests tests_DATAFILESPATH DATAFILESPATH=/home/balay/datafiles

#c89
./config/examples/asterix/asterix64-c89.py
make PETSC_ARCH=asterix64-c89 all test alltests tests_DATAFILESPATH DATAFILESPATH=/home/balay/datafiles

# basic sun tests
./config/examples/asterix/asterix64-sun.py
make PETSC_ARCH=asterix64-sun all test
./config/examples/asterix/asterix64-sun.py --with-shared-libraries=1 --with-dynamic-loading=1 -PETSC_ARCH=asterix64-sun-dynamic
make PETSC_ARCH=asterix64-sun-dynamic all alltests tests_DATAFILESPATH DATAFILESPATH=/home/balay/datafiles
./config/examples/asterix/asterix64-sun.py --with-clanguage=cxx -PETSC_ARCH=asterix64-sun-cxx  --with-debugging=0 --with-pic=0 \
--download-f-blaslapack=1 --download-spooles=1 --download-superlu=1 \
--download-superlu_dist=1 --download-hypre=1 --download-spai=1 --download-blacs=1 --download-scalapack=1 \
--download-mumps=1 --download-mpe=1 --download-sundials=1
make PETSC_ARCH=asterix64-sun-cxx all
make PETSC_ARCH=asterix64-sun-cxx CLINKER=sun-cc shared
make PETSC_ARCH=asterix64-sun-cxx test
./config/examples/asterix/asterix64-sun.py --with-shared-libraries=1 --with-dynamic-loading=1 --with-scalar-type=complex --with-clanguage=cxx -PETSC_ARCH=asterix64-sun-complex-dynamic
make PETSC_ARCH=asterix64-sun-complex-dynamic all
make PETSC_ARCH=asterix64-sun-complex-dynamic CLINKER=sun-cc shared
make PETSC_ARCH=asterix64-sun-complex-dynamic test

