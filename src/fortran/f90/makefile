# $Id: makefile,v 1.8 1998/04/23 22:50:51 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
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


