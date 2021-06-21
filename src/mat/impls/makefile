-include ../../../petscdir.mk
ALL: lib

DIRS     = dense aij shell baij adj maij kaij is sbaij normal lrc scatter blockmat composite cufft mffd transpose python submat localref nest fft elemental scalapack preallocator hypre sell dummy cdiagonal hara htool centering
LOCDIR   = src/mat/impls/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
