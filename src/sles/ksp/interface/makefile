PETSC_DIR = ../../..

ALL: lib

CFLAGS   = -I.. $(PETSC_INCLUDE) $(PCONF) \
           $(CONF) -D__DIR__='"ksp/interface/"'
SOURCEC  = itcl.c itfunc.c iguess.c itcreate.c iterativ.c itres.c itregis.c \
           xmon.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = itcl.o itfunc.o iguess.o itcreate.o iterativ.o itres.o itregis.o \
           xmon.o
LIBBASE  = libpetscksp
DIRS     =
MANSEC   = 5
LOCDIR   = src/ksp/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
