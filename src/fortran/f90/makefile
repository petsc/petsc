# $Id: makefile,v 1.18 1999/02/02 21:36:04 balay Exp balay $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = zf90vec.c zf90mat.c zf90da.c f90_nag.c f90_IRIX.c f90_rs6000.c \
           f90_t3e.c f90_solaris.c f90_hpux.c f90_win32.c
SOURCEF  = 
SOURCEH  = zf90.h f90_IRIX.h f90_rs6000.h f90_t3e.h f90_solaris.h f90_hpux.h \
	   f90_win32.h
OBJSC    = zf90vec.o zf90mat.o zf90da.o f90_nag.o f90_IRIX.o f90_rs6000.o\
           f90_t3e.o f90_solaris.o f90_hpux.o f90_win32.o
OBJSF    = 
LIBBASE  = libpetscfortran
MANSEC   = Sys
DIRS     = tests
LOCDIR   = src/fortran/f90/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base


