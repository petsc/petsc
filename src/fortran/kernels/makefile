# $Id: makefile,v 1.4 1998/03/24 17:37:40 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
FFLAGS   = 
SOURCEC  = 
SOURCEF  = fmaxpy.F fmdot.F fnorm.F fmult.F fmultadd.F fsolve.F
SOURCEH  = 
OBJSC    = 
OBJSF    = fmaxpy.o fmdot.o fnorm.o fmult.o fmultadd.o fsolve.o
LIBBASE  = libpetsckernels
MANSEC   = 7
LOCDIR   = src/fortran/kernels/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base


