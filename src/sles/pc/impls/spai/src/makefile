# $Id: makefile,v 1.8 2000/08/17 04:52:16 bsmith Exp bsmith $ 

ALL: lib
#
#    SPAI_INCLUDE should be defined in bmake/$PETSC_ARCH/base.site
#
CFLAGS   = 
FFLAGS   = 
CPPFLAGS = ${SPAI_INCLUDE}
SOURCEC  = ispai.c dspai.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ispai.o dspai.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
LOCDIR   = src/contrib/pc/spai/src/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test


