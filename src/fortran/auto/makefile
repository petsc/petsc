IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I../..  \
          $(CONF) -D__DIR__='"fortran/auto/"'
SOURCEC  = vectorf.c indexf.c da2f.c da1f.c da3f.c drawf.c \
           itfuncf.c itcreatef.c iterativf.c itclf.c \
           xmonf.c matrixf.c preconf.c pcsetf.c slesf.c\
           snesf.c snesjf.c snesmfjf.c snesutf.c\
           errf.c signalf.c fpf.c trf.c archf.c \
           comsortf.c destroyf.c plogf.c malf.c \
           cgeigf.c chebyf.c gmresf.c richf.c shellf.c shellpcf.c\
           axisf.c lgf.c gcreatef.c axpyf.c zerodiagf.c\
           vinvf.c vscatf.c  cmeshf.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vectorf.o indexf.o da2f.o da1f.o da3f.o drawf.o \
           itfuncf.o itcreatef.o iterativf.o  itclf.o  \
           xmonf.o matrixf.o preconf.o pcsetf.o slesf.o\
           snesf.o snesjf.o snesmfjf.o  snesutf.o\
           errf.o signalf.o fpf.o trf.o archf.o \
           comsortf.o destroyf.o plogf.o malf.o \
           cgeigf.o chebyf.o gmresf.o richf.o shellf.o shellpcf.o\
           axisf.o lgf.o gcreatef.o axpyf.o zerodiagf.o\
           vinvf.o vscatf.o  cmeshf.o
OBJSF    =
LIBBASE  = libpetscfortran
MANSEC   = 1

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

