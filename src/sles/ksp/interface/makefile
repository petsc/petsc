# $Id: makefile,v 1.31 1997/08/22 15:11:05 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = itcl.c itfunc.c iguess.c itcreate.c iterativ.c itres.c itregis.c \
           xmon.c eige.c dlregis.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = itcl.o itfunc.o iguess.o itcreate.o iterativ.o itres.o itregis.o \
           xmon.o eige.o dlregis.o
LIBBASE  = libpetscsles
DIRS     =
MANSEC   = 5
LOCDIR   = src/ksp/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
