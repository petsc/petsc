
ALL: lib

#requireslanguage CXXONLY

CFLAGS   = -I.
FFLAGS   =
CPPFLAGS = -DALE_HAVE_CXX_ABI -DALE_USE_LOGGING -DALE_LOGGING_USE_STAGES #-DALE_USE_DEBUGGING
SOURCEC  = ALE.cxx Coaster.cxx PreSieve.cxx Sieve.cxx Stack.cxx IndexBundle.cxx Simplicializer.cxx
SOURCEF  =
SOURCEH  = ALE.hh  Coaster.hh  PreSieve.hh  Sieve.hh  Stack.hh  IndexBundle.hh
OBJSC    = ALE.o   Coaster.o   PreSieve.o   Sieve.o   Stack.o   IndexBundle.o   Simplicializer.o
OBJSF    =
LIBBASE  = libpetscdm
MANSEC   = DA
LOCDIR   = src/dm/mesh/sieve/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
