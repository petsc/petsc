ALL: lib

CFLAGS   = -I. -I/PETSc3/ArX-2.2.4/src/boost/ ${TRIANGLE_INCLUDE}
FFLAGS   =
CPPFLAGS = -DALE_HAVE_CXX_ABI -DALE_USE_LOGGING -DALE_USE_DEBUGGING -DALE_LOGGING_USE_EVENTS #-DALE_LOGGING_USE_STAGES
SOURCEC  = ALE.cxx Coaster.cxx PreSieve.cxx Sieve.cxx Stack.cxx IndexBundle.cxx Simplicializer.cxx Mesh.cxx
SOURCEF  =
SOURCEH  = ALE.hh  Coaster.hh  PreSieve.hh  Sieve.hh  Stack.hh  IndexBundle.hh                     Mesh.hh  Sifter.hh
OBJSC    = ALE.o   Coaster.o   PreSieve.o   Sieve.o   Stack.o   IndexBundle.o   Simplicializer.o   Mesh.o
OBJSF    =
LIBBASE  = libpetscdm
MANSEC   = DA
LOCDIR   = src/dm/mesh/sieve/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
