-include ../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_H2OPUS'

SOURCECXX = math2opuscpu.cxx math2opusutilscpu.cxx
SOURCECU  = math2opus.cu math2opusutils.cu
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
LOCDIR    = src/mat/impls/h2opus/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
