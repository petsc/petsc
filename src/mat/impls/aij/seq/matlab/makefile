#requiresdefine   'PETSC_HAVE_MATLAB_ENGINE'
#requiresscalar    real
#requiresprecision double

ALL: lib

CFLAGS   = ${MATLAB_INCLUDE}
FFLAGS   =
SOURCEC  = aijmatlab.c
SOURCEF  = 
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/matlab/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
