# $Id: makefile,v 1.3 1997/09/18 16:06:47 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
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


