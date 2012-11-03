#requirespackage  'PETSC_HAVE_SIEVE'
#requireslanguage CXXONLY

ALL: lib

CPPFLAGS =
CFLAGS   =
FFLAGS   =
SOURCEC  = ALE.cxx meshcreate.c mesh.c section.c meshbardhan.c meshdolfin.c meshexodus.c meshlagrit.c meshpcice.c meshzoltan.c
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMMESH
LOCDIR   = src/dm/impls/mesh/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
