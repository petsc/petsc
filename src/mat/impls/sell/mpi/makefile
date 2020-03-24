-include ../petscdir.mk
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC	 = mpisell.c mmsell.c
SOURCEF	 =
SOURCEH	 = mpisell.h
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/sell/mpi/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
