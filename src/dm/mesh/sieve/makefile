ALL: lib

CFLAGS   =
FFLAGS   =
CPPFLAGS =
SOURCEC  = ALE.cxx
SOURCEF  =
SOURCEH  = ALE_options.hh Filter.hh Hierarchy.hh Hierarchy_New.hh Map.hh Meshing.hh MeshSurgery.hh X.hh XSieve.hh XSifter.hh
OBJSC    = ALE.o               
OBJSF    =
LIBBASE  = libpetscdm
MANSEC   = Mesh
LOCDIR   = src/dm/mesh/sieve/

ALE.o:  ALE.hh 
ALE.hh: ALE_mem.hh ALE_log.hh ALE_containers.hh 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
