#PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = itcl.c itfunc.c iguess.c itcreate.c iterativ.c itres.c itregis.c \
           xmon.c eige.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = itcl.o itfunc.o iguess.o itcreate.o iterativ.o itres.o itregis.o \
           xmon.o eige.o
LIBBASE  = libpetscksp
DIRS     =
MANSEC   = 5
LOCDIR   = src/ksp/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
