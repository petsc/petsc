IPETSCDIR = ../../../

ALL: lib 
CFLAGS    = $(OPT) $(INCLUDEDIRS) -I$(IPETSCDIR)/pinclude $(CONF)\
            -D__DIR__='"sys/src/"'
SOURCEC   = adebug.c  err.c signal.c fp.c  tr.c arch.c  file.c  nreg.c \
            mpiu.c comsort.c destroy.c plog.c options.c mal.c
SOURCEF   =
SOURCEH   =
OBJSC     = adebug.o  err.o signal.o fp.o  tr.o arch.o file.o  nreg.o \
            mpiu.o comsort.o destroy.o  plog.o options.o mal.o
OBJSF     =
WSOURCEC  = errorf.c  
WOBJS     = errorf.o 
MANSEC    = 7

LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = 

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
