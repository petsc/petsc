
#requirespackage   'PETSC_HAVE_FFTW'
#requiresprecision double

ALL: lib

# Set EXTRA_FLAGS="-DPETSC_DEBUG_USFFT" to turn on debugging output
CFLAGS   = ${EXTRA_FLAGS}
FFLAGS   =
SOURCEC  = matusfft.c
OBJSC    = matusfft.o
LIBBASE  = libpetscdm
MANSEC   = DA
DIRS     =
LOCDIR   = src/dm/da/src/usfft

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test
