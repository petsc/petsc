PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I../..  $(PCONF)\
          $(CONF) -D__DIR__='"fortran/auto/"'

SOURCEC  = archf.c axisf.c axpyf.c bjacobif.c cgeigf.c \
           chebyf.c cmeshf.c comsortf.c da2f.c destroyf.c \
           drawf.c eisenf.c fpf.c gmresf.c iccbsf.c \
           iluf.c indexf.c itclf.c itcreatef.c itfuncf.c \
           itresf.c lgf.c luf.c matrixf.c mgf.c mgfuncf.c \
           pcsetf.c preconf.c  richf.c signalf.c slesf.c \
           snesf.c snesutf.c sorf.c stridef.c trf.c \
           umlsf.c umtrf.c vectorf.c vinvf.c vscatf.c \
           zerodiagf.c
SOURCEF  =
SOURCEH  = 
OBJSC    = archf.o axisf.o axpyf.o bjacobif.o cgeigf.o \
           chebyf.o cmeshf.o comsortf.o da2f.o destroyf.o \
           drawf.o eisenf.o fpf.o gmresf.o iccbsf.o \
           iluf.o indexf.o itclf.o itcreatef.o itfuncf.o \
           itresf.o lgf.o luf.o matrixf.o mgf.o mgfuncf.o \
           pcsetf.o preconf.o  richf.o signalf.o slesf.o \
           snesf.o snesutf.o sorf.o stridef.o trf.o \
           umlsf.o umtrf.o vectorf.o vinvf.o vscatf.o \
           zerodiagf.o
OBJSF    =
LIBBASE  = libpetscfortran
MANSEC   = 1

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

