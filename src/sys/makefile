PETSC_DIR = ../..

ALL: lib 
CFLAGS    =  $(PETSC_INCLUDE) $(CONF) -D__DIR__='"sys/"' $(PCONF)
SOURCEC   =
SOURCEF   =
SOURCEH   = nreg.h
OBJSC     =
OBJSF     =
LIBBASE   = libpetscsys
DIRS      = src examples

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
