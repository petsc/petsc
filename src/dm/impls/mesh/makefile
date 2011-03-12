#requirespackage  'PETSC_HAVE_SIEVE'
#requireslanguage CXXONLY

ALL: lib

CPPFLAGS =
CFLAGS   =
FFLAGS   =
SOURCEC  = ALE.cxx mesh.c meshbardhan.c meshpcice.c meshpylith.c meshlagrit.c meshzoltan.c meshdolfin.c meshexodus.c section.c cartesian.c
SOURCEF  =
SOURCEH  =
OBJSC    = ALE.o   mesh.o meshbardhan.o meshpcice.o meshpylith.o meshlagrit.o meshzoltan.o meshdolfin.o meshexodus.o section.o cartesian.o
OBJSF    =
DIRS     =
LIBBASE  = libpetscdm
MANSEC   = DM
LOCDIR   = src/dm/impls/mesh/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
