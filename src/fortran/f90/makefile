# $Id: makefile,v 1.25 2000/10/06 15:32:49 bsmith Exp balay $ 

#requirespackage 'PETSC_HAVE_F90_H'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = zf90vec.c zf90mat.c zf90da.c zf90ksp.c zf90snes.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = zf90vec.o zf90mat.o zf90da.o zf90ksp.o zf90snes.o
OBJSF    = 
LIBBASE  = libpetscfortran
MANSEC   = Sys
DIRS     = tests
LOCDIR   = src/fortran/f90/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


