PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I../..  $(PCONF)\
          $(CONF) -D__DIR__='"fortran/auto/"'
SOURCEC  = vectorf.c indexf.c da2f.c drawf.c \
           itfuncf.c itcreatef.c itclf.c \
           matrixf.c preconf.c pcsetf.c slesf.c\
           snesf.c snesutf.c stridef.c\
           signalf.c fpf.c trf.c archf.c \
           comsortf.c destroyf.c  \
           cgeigf.c chebyf.c gmresf.c richf.c \
           axisf.c lgf.c axpyf.c zerodiagf.c\
           vinvf.c vscatf.c  cmeshf.c iluf.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vectorf.o indexf.o da2f.o drawf.o \
           itfuncf.o itcreatef.o itclf.o  \
           matrixf.o preconf.o pcsetf.o slesf.o\
           snesf.o snesutf.o stridef.o\
           signalf.o fpf.o trf.o archf.o \
           comsortf.o destroyf.o \
           cgeigf.o chebyf.o gmresf.o richf.o \
           axisf.o lgf.o axpyf.o zerodiagf.o\
           vinvf.o vscatf.o  cmeshf.o iluf.o
OBJSF    =
LIBBASE  = libpetscfortran
MANSEC   = 1

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

