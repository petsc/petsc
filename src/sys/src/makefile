IPETSCDIR = ../../../

ALL: lib 
CFLAGS    = $(OPT) $(INCLUDEDIRS) -I$(IPETSCDIR)/pinclude $(CONF)\
            -D__DIR__='"sys/src/"'
SOURCEC   = adebug.c err.c signal.c fp.c tr.c arch.c file.c nreg.c \
            mpiu.c comsort.c destroy.c plog.c options.c mal.c
SOURCEF   =
SOURCEH   =
OBJSC     = adebug.o err.o signal.o fp.o tr.o arch.o file.o nreg.o \
            mpiu.o comsort.o destroy.o  plog.o options.o mal.o
OBJSF     =
WSOURCEC  = adebugf.c errf.c signalf.c fpf.c trf.c archf.c filef.c nregf.c \
            mpiuf.c comsortf.c destroyf.c plogf.c optionsf.c malf.c
WOBJS     = adebugf.o  errf.o signalf.o fpf.o trf.o archf.o filef.o nregf.o \
            mpiuf.o comsortf.o destroyf.o plogf.o optionsf.o malf.o
MANSEC    = 7

LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = 

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
