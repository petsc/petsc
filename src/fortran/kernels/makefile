# $Id: makefile,v 1.14 1999/03/01 04:59:16 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = 
SOURCEF  = fmaxpy.F fmdot.F fnorm.F fmult.F fmultadd.F fsolve.F fsolvebaij.F \
           sgemv.F
SOURCEH  = 
OBJSC    = 
OBJSF    = fmaxpy.o fmdot.o fnorm.o fmult.o fmultadd.o fsolve.o fsolvebaij.o \
           sgemv.o
DOCS     = readme
LIBBASE  = libpetsc
MANSEC   = Sys
LOCDIR   = src/fortran/kernels/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test


