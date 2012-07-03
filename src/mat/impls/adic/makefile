
#requirespackage  'PETSC_HAVE_ADIC'
#requiresscalar real 
#requiresprecision double

ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCEC  = matadic.c nladic.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = matadic.o nladic.o
OBJSF    = 
LIBBASE  = libpetscsnes
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/adic/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
