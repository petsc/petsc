# $Id: makefile,v 1.129 2001/09/25 14:36:11 balay Exp $ 

ALL: lib


CFLAGS   = -DPETSC_FORTRAN_STUBS
FFLAGS   =
SOURCEF	 = 
SOURCEH	 = 
OBJSC	 = ${SOURCEC:.c=.o}
OBJSF	 =
LIBBASE	 = libpetscfortran
MANSEC	 = Sys
LOCDIR   = src/fortran/auto/

include makefile.src
include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

