# $Id: makefile,v 1.15 1998/09/25 19:12:02 balay Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = zf90vec.c zf90mat.c zf90da.c f90_nag.c f90_IRIX.c f90_rs6000.c \
           f90_t3e.c f90_solaris.c f90_hpux.c
SOURCEF  = 
SOURCEH  = zf90.h f90_IRIX.h f90_rs6000.h f90_t3e.h f90_solaris.h f90_hpux.h
OBJSC    = zf90vec.o zf90mat.o zf90da.o f90_nag.o f90_IRIX.o f90_rs6000.o\
           f90_t3e.o f90_solaris.o f90_hpux.o
OBJSF    = 
LIBBASE  = libpetscfortran
MANSEC   = 7
DIRS     = test
LOCDIR   = src/fortran/f90/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base


