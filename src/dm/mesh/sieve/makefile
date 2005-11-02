
ALL: lib

#requireslanguage CXXONLY

CFLAGS   = -I.
FFLAGS   =
CPPFLAGS = 
SOURCEC  = ALE.cxx PreSieve.cxx Sieve.cxx Stack.cxx IndexBundle.cxx Simplicializer.cxx
SOURCEF  =
SOURCEH  = ALE.hh  PreSieve.hh  Sieve.hh  Stack.hh  IndexBundle.hh
OBJSC    = ALE.o   PreSieve.o   Sieve.o   Stack.o   IndexBundle.o   Simplicializer.o
OBJSF    =
LIBBASE  = libpetscdm
MANSEC   = DA
LOCDIR   = src/dm/mesh/sieve/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
