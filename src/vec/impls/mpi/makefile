#
#   I'd like to add the dependancies pvec.c depends on pdvec.c ...
#  but can't figure out how.
#
PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I.. -I../.. \
           $(CONF) $(COPT) $(PCONF) \
           -D__DIR__='"vec/impls/mpi/"'
SOURCEC  = pbvec.c
SOURCEF  =
SOURCEH  = pvecimpl.h pdvec.c pvec2.c
OBJSC    = pbvec.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/impls/mpi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

