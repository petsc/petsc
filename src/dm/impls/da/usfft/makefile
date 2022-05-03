-include ../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_FFTW'
#requiresprecision double

# Set CFLAGS="-DPETSC_DEBUG_USFFT" to turn on debugging output
SOURCEC  = matusfft.c
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMDA
DIRS     =
LOCDIR   = src/dm/impls/da/usfft/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
