#PETSC_DIR = ../../..

ALL: lib

CFLAGS   =  $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = gcreate.c convert.c stash.c axpy.c zerodiag.c matio.c matioall.c \
           conaij.c conbdiag.c
SOURCEF  =
SOURCEH  =
OBJSC    = gcreate.o convert.o stash.o axpy.o zerodiag.o matio.o matioall.o \
           conaij.o conbdiag.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/utils/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
