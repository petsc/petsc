#requirespackage  'PETSC_HAVE_SIEVE'
#requireslanguage CXXONLY

ALL: lib

CPPFLAGS = ${SIEVE_FLAGS}
CFLAGS   = -Isieve ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE}
FFLAGS   =
SOURCEC  = mesh.c meshpcice.c meshpylith.c section.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = mesh.o meshpcice.o meshpylith.o section.o
OBJSF    =
DIRS     = sieve
LIBBASE  = libpetscdm
MANSEC   = DA
LOCDIR   = src/dm/mesh/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
