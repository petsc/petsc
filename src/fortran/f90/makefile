#PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
FFLAGS   = 
SOURCEC  = zf90.c f90_nag.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = zf90.o f90_nag.o
OBJSF    = 
LIBBASE  = libpetscfortran
MANSEC   = 7
LOCDIR   = src/fortran/f90/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base


