# $Id: makefile,v 1.30 2000/08/17 04:51:22 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  =
SOURCEF  =
SOURCEH  = matimpl.h ../../include/petscmat.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = interface impls examples utils matfd partition order color
LOCDIR   = src/mat/
MANSEC   = Mat

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
