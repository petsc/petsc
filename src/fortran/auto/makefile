IPETSCDIR = ..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_DIR) -I..  \
          -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"c2f77/"'
SOURCEC  = vectorf.c indexf.c da2f.c da1f.c da3f.c drawf.c \
           itfuncf.c itcreatef.c iterativf.c itresf.c itclf.c itregisf.c \
           xmonf.c matrixf.c preconf.c pcsetf.c pcregisf.c slesf.c\
           snesf.c snesjf.c snesmfjf.c snesregif.c snesutf.c\
           adebugf.c errf.c signalf.c fpf.c trf.c archf.c \
           comsortf.c destroyf.c plogf.c malf.c vectorf.c\
           cgeigf.c chebyf.c gmresf.c richf.c shellf.c shellpcf.c\
           pbvecf.c bvec2f.c axisf.c lgf.c gcreatef.c axpyf.c zerodiagf.c\
           vinvf.c gcreatevf.c vscatf.c  cmeshf.c sregisf.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vectorf.o indexf.o da2f.o da1f.o da3f.o drawf.o \
           itfuncf.o itcreatef.o iterativf.o itresf.o itclf.o itregisf.o \
           xmonf.o matrixf.o preconf.o pcsetf.o pcregisf.o slesf.o\
           snesf.o snesjf.o snesmfjf.o snesregif.o snesutf.o\
           adebugf.o errf.o signalf.o fpf.o trf.o archf.o \
           comsortf.o destroyf.o plogf.o malf.o vectorf.o\
           cgeigf.o chebyf.o gmresf.o richf.o shellf.o shellpcf.o\
           pbvecf.o bvec2f.o axisf.o lgf.o gcreatef.o axpyf.o zerodiagf.o\
           vinvf.o gcreatevf.o vscatf.o  cmeshf.o sregisf.o
OBJSF    =
LIBBASE  = libpetscf77
MANSEC   = 1

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

