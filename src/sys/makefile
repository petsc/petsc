IPETSCDIR = ../..

ALL: lib 
CFLAGS    = $(OPT) $(PETSC_INCLUDE) $(CONF) -D__DIR__='"sys/"' 
SOURCEC   =
SOURCEF   =
SOURCEH   = nreg.h
OBJSC     =
OBJSF     =
MANSEC    = 7
LIBBASE   = libpetscsys
DIRS      = src examples

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
