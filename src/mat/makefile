# $Id: makefile,v 1.31 2000/11/28 17:28:44 bsmith Exp balay $ 

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
