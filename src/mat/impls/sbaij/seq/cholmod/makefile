-include ../petscdir.mk
#requirespackage    'PETSC_HAVE_SUITESPARSE'
#requiresprecision  double

ALL: lib

CFLAGS	 = ${SUITESPARSE_INCLUDE}
FFLAGS	 =
CPPFLAGS =
FPPFLAGS =
SOURCEC	 = sbaijcholmod.c
SOURCEF	 =
SOURCEH	 = cholmodimpl.h
LIBBASE	 = libpetscmat
DIRS     =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/sbaij/seq/cholmod/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
