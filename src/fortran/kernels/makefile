# $Id: makefile,v 1.1 1997/09/12 19:25:06 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
FFLAGS   = 
SOURCEC  = 
SOURCEF  = fmaxpy.F fmdot.F fnorm.F fmult.F
SOURCEH  = 
OBJSC    = 
OBJSF    = fmaxpy.o fmdot.o fnorm.o fmult.o
LIBBASE  = libpetsckernels
MANSEC   = 7
LOCDIR   = src/fortran/kernels/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base


