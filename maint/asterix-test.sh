#!/bin/sh
set -x

# basic gcc tests
./config/asterix.py
make PETSC_ARCH=asterix all test
make PETSC_ARCH=asterix testexamples testfortran
./config/asterix.py --with-clanguage=cxx -PETSC_ARCH=asterix-cxx-opt --with-debugging=0
make PETSC_ARCH=asterix-cxx-opt all test
./config/asterix.py --with-scalar-type=complex -PETSC_ARCH=asterix-complex
make PETSC_ARCH=asterix-complex all test

./config/asterix.py --download-prometheus=1 --download-parmetis=1 -PETSC_ARCH=asterix-prometheus
make PETSC_ARCH=asterix-prometheus all test

# basic intel tests
./config/asterix-intel.py --with-debugging=0 -PETSC_ARCH=asterix-intel-opt --download-spooles=1 --download-superlu=1 \
--download-superlu_dist=1 --download-hypre=1 --download-spai=1 --download-blacs=1 --download-scalapack=1 \
--download-mumps=1 --download-mpe=1 --download-sundials=1 LIBS=/usr/lib/libm.a
make PETSC_ARCH=asterix-intel-opt all test
./config/asterix-intel.py --with-clanguage=cxx -PETSC_ARCH=asterix-intel-cxx
make PETSC_ARCH=asterix-intel-cxx all test
./config/asterix-intel.py --with-scalar-type=complex -PETSC_ARCH=asterix-intel-complex
make PETSC_ARCH=asterix-intel-complex all test

./config/asterix-intel.py --download-prometheus=1 --download-parmetis=1 --with-clanguage=cxx -PETSC_ARCH=asterix-intel-cxx-prometheus
make PETSC_ARCH=asterix-intel-cxx-prometheus all test

# tops test
./config/asterix-tops.py
make PETSC_ARCH=asterix-tops all test

# basic sun tests
./config/asterix-sun.py
make PETSC_ARCH=asterix-sun all test
./config/asterix-sun.py --with-shared=1 --with-dynamic=1 -PETSC_ARCH=asterix-sun-dynamic
make PETSC_ARCH=asterix-sun-dynamic all test
./config/asterix-sun.py --with-clanguage=cxx -PETSC_ARCH=asterix-sun-cxx  --with-debugging=0 \
--download-f-blaslapack=1 --download-spooles=1 --download-superlu=1 \
--download-superlu_dist=1 --download-hypre=1 --download-spai=1 --download-blacs=1 --download-scalapack=1 \
--download-mumps=1 --download-mpe=1 --download-sundials=1
make PETSC_ARCH=asterix-sun-cxx all
make PETSC_ARCH=asterix-sun-cxx CLINKER=sun-cc shared
make PETSC_ARCH=asterix-sun-cxx test
./config/asterix-sun.py --with-shared=1 --with-dynamic=1 --with-scalar-type=complex -PETSC_ARCH=asterix-sun-complex-dynamic
make PETSC_ARCH=asterix-sun-complex-dynamic all
make PETSC_ARCH=asterix-sun-complex-dynamic CLINKER=sun-cc shared
make PETSC_ARCH=asterix-sun-complex-dynamic test

# basic gcc4 tests
./config/asterix-gcc4.py
make PETSC_ARCH=asterix-gcc4 all test
./config/asterix-gcc4.py --with-clanguage=cxx --with-sieve=1 -PETSC_ARCH=asterix-gcc4-cxx
make PETSC_ARCH=asterix-gcc4-cxx all test
./config/asterix-gcc4.py --with-scalar-type=complex -PETSC_ARCH=asterix-gcc4-complex-opt --with-debugging=0
make PETSC_ARCH=asterix-gcc4-complex-opt all test

