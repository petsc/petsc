# $Id: makefile,v 1.18 2001/08/22 18:03:08 balay Exp $ 

# This directory contains some computational kernels written in
# Fortran77.  These are mainly intended to provide high performance for
# COMPLEX arithmetic (since most C++ compilers do not do a good job
# optimizing complex number calculations), but can also be used to
# compare against C versions of the same kernels.

ALL: lib
SPECIALFASTLIB = yes
specialfastlib: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = 
SOURCEF  = fmaxpy.F fmdot.F fnorm.F fmult.F fmultadd.F fsolve.F fsolvebaij.F \
           sgemv.F frelax.F fwaxpy.F faypx.F fcopy.F
SOURCEH  = 
OBJSC    = 
OBJSF    = fmaxpy.o fmdot.o fnorm.o fmult.o fmultadd.o fsolve.o fsolvebaij.o \
           sgemv.o frelax.o fwaxpy.o faypx.o fcopy.o
LIBBASE  = libpetsc
MANSEC   = Sys
LOCDIR   = src/fortran/kernels/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


