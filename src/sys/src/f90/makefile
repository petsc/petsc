# $Id: makefile,v 1.22 2000/08/17 04:53:44 bsmith Exp $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = f90_nag.c f90_IRIX.c f90_rs6000.c f90_t3e.c f90_solaris.c f90_hpux.c \
	   f90_win32.c f90_alpha.c f90_solaris_old.c
SOURCEF  = 
SOURCEH  = f90impl.h f90_IRIX.h f90_rs6000.h f90_t3e.h f90_solaris.h f90_hpux.h \
	   f90_win32.h f90_alpha.h f90_solaris_old.h
OBJSC    = f90_nag.o f90_IRIX.o f90_rs6000.o f90_t3e.o f90_solaris.o f90_hpux.o \
	   f90_win32.o f90_alpha.o f90_solaris_old.o
OBJSF    = 
LIBBASE  = libpetscsys
MANSEC   = Sys
DIRS     = 
LOCDIR   = src/sys/src/f90/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test


