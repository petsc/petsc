#!/bin/sh
set -x
ECHO=

OPT1='--with-shared-libraries=1 --with-dynamic-loading=1 --with-debugging=0'
OPT2='--with-clanguage=cxx --with-sieve=1 --with-log=0'
OPT3='--with-scalar-type=complex --with-clanguage=cxx --with-pic=0 --with-log=0'

DATAFILESPATH=/home/petsc/datafiles

PACKAGES='--download-mpich=1 --download-plapack=1
--download-parmetis=1 --download-triangle=1
--download-spooles=1 --download-superlu=1 --download-superlu_dist=1
--download-blacs=1 --download-scalapack=1 --download-mumps=1
--download-mpe=1 --download-fftw'

PKG1='--download-spai=1 --download-chaco=1 --download-sundials=1 --download-umfpack=1
--download-hypre=1 --download-prometheus=1 --download-hdf5=1 '

PKG2='--download-boost=1  --download-tetgen=1'

PKG3=''

# missing package tests:
# party - ?
# scotch - binary?
# jostle - binary?
# dscpack - download does not work
# cblas/fblas ??

GNUCOMP='CC=gcc FC=gfortran CXX=g++'
INTELCOMP='CC=icc FC=ifort CXX=icpc'
SUNCMP='CC=sun-cc FC=sun-f90 CXX=sun-CC'

BUILD='all test testexamples testfortran'


# Gnu compilers
${ECHO} ./config/configure.py PETSC_ARCH=package-gnu-opt1 ${BASIC} ${PACKAGES} ${PKG1} ${OPT1} ${GNUCOMP}
${ECHO} make PETSC_ARCH=package-gnu-opt1 ${BUILD}

${ECHO} ./config/configure.py PETSC_ARCH=package-gnu-opt2 ${BASIC} ${PACKAGES} ${PKG2} ${OPT2} ${GNUCOMP}
${ECHO} make PETSC_ARCH=package-gnu-opt2 ${BUILD}

${ECHO} ./config/configure.py PETSC_ARCH=package-gnu-opt3 ${BASIC} ${PACKAGES} ${PKG3} ${OPT3} ${GNUCOMP}
${ECHO} make PETSC_ARCH=package-gnu-opt3 ${BUILD}

# Intel compilers
${ECHO} ./config/configure.py PETSC_ARCH=package-intel-opt1 ${BASIC} ${PACKAGES} ${PKG1} ${OPT1} ${INTELCOMP}
${ECHO} make PETSC_ARCH=package-intel-opt1 ${BUILD}

${ECHO} ./config/configure.py PETSC_ARCH=package-intel-opt2 ${BASIC} ${PACKAGES} ${PKG2} ${OPT2} ${INTELCOMP}
${ECHO} make PETSC_ARCH=package-intel-opt2 ${BUILD}

${ECHO} ./config/configure.py PETSC_ARCH=package-intel-opt3 ${BASIC} ${PACKAGES} ${PKG3} ${OPT3} ${INTELCOMP}
${ECHO} make PETSC_ARCH=package-intel-opt3 ${BUILD}

# Sun compilers
${ECHO} ./config/configure.py PETSC_ARCH=package-sun-opt1 ${BASIC} ${PACKAGES} ${PKG1} ${OPT1} ${SUNCOMP}
${ECHO} make PETSC_ARCH=package-sun-opt1 ${BUILD}

${ECHO} ./config/configure.py PETSC_ARCH=package-sun-opt2 ${BASIC} ${PACKAGES} ${PKG2} ${OPT2} ${SUNCOMP}
${ECHO} make PETSC_ARCH=package-sun-opt2 ${BUILD}

${ECHO} ./config/configure.py PETSC_ARCH=package-sun-opt3 ${BASIC} ${PACKAGES} ${PKG3} ${OPT3} ${SUNCOMP}
${ECHO} make PETSC_ARCH=package-sun-opt3 ${BUILD}
