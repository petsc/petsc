# $Id: makefile,v 1.44 2001/08/22 18:00:29 balay Exp $ 
#requirespackage    'PETSC_HAVE_SPOOLES'
#requiresprecision  double

ALL: lib

CFLAGS   = ${SPOOLES_INCLUDE}
FFLAGS   =
SOURCEC	 = mpiaijspooles.c mpispooles.c
SOURCEF	 =
SOURCEH	 =
OBJSC	 = mpiaijspooles.o mpispooles.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/spooles/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
