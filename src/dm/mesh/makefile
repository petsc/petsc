#requirespackage  'PETSC_HAVE_SIEVE'
#requireslanguage CXXONLY

ALL: lib

CPPFLAGS = -DALE_USE_DEBUGGING  -DALE_HAVE_CXX_ABI -DALE_USE_LOGGING -DALE_LOGGING_USE_EVENTS -DALE_LOGGING_LOG_MEM #-DALE_LOGGING_USE_STAGES
CFLAGS   = -Isieve ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE}
FFLAGS   =
SOURCEC  = mesh.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = mesh.o
OBJSF    =
DIRS     = sieve
LIBBASE  = libpetscdm
MANSEC   = DA
LOCDIR   = src/dm/mesh/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
