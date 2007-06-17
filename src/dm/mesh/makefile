#requirespackage  'PETSC_HAVE_SIEVE'
#requireslanguage CXXONLY

ALL: lib

CPPFLAGS = ${SIEVE_FLAGS}
CFLAGS   = -Isieve ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE} ${ZOLTAN_INCLUDE}
FFLAGS   =
SOURCEC  = mesh.c meshpcice.c meshpylith.c meshzoltan.c meshdolfin.c section.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = mesh.o meshpcice.o meshpylith.o meshzoltan.o meshdolfin.o section.o
OBJSF    =
DIRS     = sieve impls
LIBBASE  = libpetscdm
MANSEC   = DA
LOCDIR   = src/dm/mesh/

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test
