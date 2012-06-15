#requirespackage 'PETSC_HAVE_ELEMENTAL'

ALL: lib

FFLAGS   =
SOURCEC  = matelem.cxx
SOURCEF  =
SOURCEH  = matelemimpl.h
OBJSC    = $(SOURCEC:.cxx=.o)
OBJSF    =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/elemental/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
