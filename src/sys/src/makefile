IPETSCDIR = ../../../

ALL: lib 
CFLAGS    = $(OPT) $(INCLUDEDIRS) -I$(IPETSCDIR)/pinclude $(CONF)\
            -D__DIR__='"sys/error/"'
SOURCEC   = adebug.c  error.c signal.c fp.c
SOURCEF   =
SOURCEH   =
OBJSC     = adebug.o  error.o signal.o fp.o
OBJSF     =
WSOURCEC  = errorf.c  
WOBJS     = errorf.o 
MANSEC    = 7

LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = 

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
