
#requirespackage    'PETSC_HAVE_CHOLMOD'
#requiresprecision  double

ALL: lib

CFLAGS	 = ${CHOLMOD_INCLUDE}
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

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
