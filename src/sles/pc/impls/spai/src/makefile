# $Id: makefile,v 1.10 2001/08/22 18:01:32 balay Exp $ 

ALL: lib
#
#    SPAI_INCLUDE should be defined in bmake/$PETSC_ARCH/packages
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
LOCDIR   = src/sles/pc/impls/spai/src/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


