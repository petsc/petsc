# $Id: makefile,v 1.39 2001/08/22 17:59:55 balay Exp $

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = map.c vector.c veccreate.c vecreg.c vecregall.c dlregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ${SOURCEC:.c=.o}
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/interface/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
