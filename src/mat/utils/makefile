# $Id: makefile,v 1.32 1997/08/22 15:15:22 bsmith Exp bsmith $

ALL: lib

CFLAGS   =  $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = gcreate.c convert.c stash.c axpy.c zerodiag.c matio.c matioall.c \
           conaij.c conbdiag.c getcolv.c
SOURCEF  =
SOURCEH  =
OBJSC    = gcreate.o convert.o stash.o axpy.o zerodiag.o matio.o matioall.o \
           conaij.o conbdiag.o getcolv.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/utils/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
