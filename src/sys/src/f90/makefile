# $Id: makefile,v 1.2 2000/09/06 22:57:36 balay Exp balay $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = f90_common.c
SOURCEF  = 
SOURCEH  = f90_nag.c f90_IRIX.c f90_rs6000.c f90_t3e.c f90_solaris.c f90_hpux.c \
	   f90_win32.c f90_alpha.c f90_solaris_old.c f90_absoft.c
OBJSC    = f90_common.o
OBJSF    = 
LIBBASE  = libpetsc
MANSEC   = Sys
DIRS     = 
LOCDIR   = src/sys/src/f90/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test


