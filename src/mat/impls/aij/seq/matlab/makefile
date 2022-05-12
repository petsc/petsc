-include ../../../../../../petscdir.mk
#requiresdefine   'PETSC_HAVE_MATLAB_ENGINE'

CFLAGS   = ${MATLAB_INCLUDE}
SOURCEC  = aijmatlab.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/matlab/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
