# $Id: makefile,v 1.44 2001/08/22 18:00:29 balay Exp $ 
#requirespackage   'PETSC_HAVE_MUMPS'
#requiresprecision double

ALL: lib

CFLAGS   = ${MUMPS_INCLUDE}
FFLAGS   =
SOURCEC	 = mumps.c
SOURCEF	 =
SOURCEH	 =
OBJSC	 = mumps.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/mumps

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
