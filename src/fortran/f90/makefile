# $Id: makefile,v 1.3 1998/03/17 22:15:17 bsmith Exp balay $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
FFLAGS   = 
SOURCEC  = zf90.c f90_nag.c f90_IRIX.c
SOURCEF  = f90_IRIX.h
SOURCEH  = 
OBJSC    = zf90.o f90_nag.o f90_IRIX.o
OBJSF    = 
LIBBASE  = libpetscfortran
MANSEC   = 7
DIR      = test
LOCDIR   = src/fortran/f90/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base


