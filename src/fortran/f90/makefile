# $Id: makefile,v 1.5 1998/03/17 22:56:25 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
FFLAGS   = 
SOURCEC  = zf90.c f90_nag.c f90_IRIX.c f90_rs6000.c
SOURCEF  = 
SOURCEH  = f90_IRIX.h f90_rs6000.h
OBJSC    = zf90.o f90_nag.o f90_IRIX.o f90_rs6000.o
OBJSF    = 
LIBBASE  = libpetscfortran
MANSEC   = 7
DIRS     = test
LOCDIR   = src/fortran/f90/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base


