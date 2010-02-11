#requirespackage  'PETSC_HAVE_SIEVE'
#requireslanguage CXXONLY

ALL: lib

CPPFLAGS =
CFLAGS   =
FFLAGS   =
SOURCEC  = mesh.c meshbardhan.c meshpcice.c meshpylith.c meshlagrit.c meshzoltan.c meshdolfin.c meshexodus.c section.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = mesh.o meshbardhan.o meshpcice.o meshpylith.o meshlagrit.o meshzoltan.o meshdolfin.o meshexodus.o section.o
OBJSF    =
DIRS     = sieve impls
LIBBASE  = libpetscdm
MANSEC   = Mesh
LOCDIR   = src/dm/mesh/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
