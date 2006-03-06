ALL: lib

CFLAGS   = -I. ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE}
FFLAGS   =
CPPFLAGS = -DALE_USE_DEBUGGING -DALE_HAVE_CXX_ABI -DALE_USE_LOGGING -DALE_LOGGING_USE_EVENTS #-DALE_LOGGING_USE_STAGES
SOURCEC  = ALE.cxx
SOURCEF  =
SOURCEH  = ALE.hh  Coaster.hh  Mesh.hh  Sifter.hh
OBJSC    = ALE.o               
OBJSF    =
LIBBASE  = libpetscdm
MANSEC   = DA
LOCDIR   = src/dm/mesh/sieve/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
