#/bin/bash
export PETSC_DIR=/home/balay/petsc-dl
export PETSC_ARCH=asterix-cxx
./config/asterix.py --with-clanguage=cxx -PETSC_ARCH=asterix-cxx
make PETSC_ARCH=asterix allfortranstubs
make
make testexamples testfortran
make ACTION=testexamples_Fortran_NoComplex tree DATAFILESPATH=/home/balay/datafiles

