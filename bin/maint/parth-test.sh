#!/bin/sh
set -x

# basic cygwin-ms test
./config/cygwin-ms.py --with-clanguage=cxx --with-debugging=0 PETSC_ARCH=cygwin-ms-cxx
make PETSC_ARCH=cygwin-ms-cxx all alltests DATAFILESPATH=/home/balay/datafiles 

# basic cygwin-borland test
./config/cygwin-borland.py
make PETSC_ARCH=cygwin-borland all test

# basic cygwin gnu test
./config/cygwin.py
make PETSC_ARCH=cygwin all test

# other cygwin-ms tests
./config/cygwin-ms.py
make PETSC_ARCH=cygwin-ms all alltests DATAFILESPATH=/home/balay/datafiles 

./config/cygwin-ms.py --with-clanguage=cxx --with-scalar-type=complex --with-debugging=0 PETSC_ARCH=cygwin-ms-cxx-complex
make PETSC_ARCH=cygwin-ms-cxx-complex  all alltests DATAFILESPATH=/home/balay/datafiles 

