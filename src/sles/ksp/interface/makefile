IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) -I.. $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"ksp/src/"'
SOURCEC  = itcl.c itfunc.c iguess.c itcreate.c iterativ.c itres.c itregis.c \
           xmon.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = itcl.o itfunc.o iguess.o itcreate.o iterativ.o itres.o itregis.o \
           xmon.o
LIBBASE  = libpetscksp
DIRS     =
MANSEC   = 5

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
