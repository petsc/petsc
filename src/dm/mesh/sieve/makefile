ALL: lib

CFLAGS   = -I. ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE}
FFLAGS   =
CPPFLAGS = ${SIEVE_FLAGS}
SOURCEC  = ALE.cxx
SOURCEF  =
SOURCEH  = ALE_containers.hh ALE_exception.hh ALE_log.hh ALE_mem.hh ALE.hh CartesianSieve.hh Completion.hh \
           Distribution.hh Field.hh Filter.hh Generator.hh Map.hh Mesh.hh Numbering.hh Partitioner.hh \
           SectionCompletion.hh Sections.hh Sieve.hh SieveAlgorithms.hh Sifter.hh XSifter.hh
OBJSC    = ALE.o               
OBJSF    =
LIBBASE  = libpetscdm
MANSEC   = DA
LOCDIR   = src/dm/mesh/sieve/

ALE.o:  ALE.hh 
ALE.hh: ALE_mem.hh ALE_log.hh ALE_containers.hh 

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test
