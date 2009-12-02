#requirespackage  'PETSC_HAVE_SIEVE'
#requireslanguage CXXONLY

ALL: lib

CPPFLAGS = -DALE_USE_DEBUGGING  -DALE_HAVE_CXX_ABI -DALE_USE_LOGGING -DALE_LOGGING_USE_EVENTS #-DALE_LOGGING_USE_STAGES
CFLAGS   = -I../../../../dm/mesh/sieve ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE}
FFLAGS   =
SOURCEC  = sieveVec.c
SOURCEF  =
SOURCEH  =
OBJSC    = sieveVec.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/sieve/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
