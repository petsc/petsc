# $Id: makefile,v 1.7 1999/11/05 14:48:08 bsmith Exp bsmith $ 

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
LIBBASE  = libpetscslesspai
DIRS     = 
LOCDIR   = src/contrib/pc/spai/src/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test


