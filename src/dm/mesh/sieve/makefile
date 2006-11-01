ALL: lib

CFLAGS   = -I. ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE}
FFLAGS   =
CPPFLAGS = ${SIEVE_FLAGS}
SOURCEC  = ALE.cxx
SOURCEF  =
SOURCEH  = ALE_containers.hh ALE_exception.hh ALE.hh ALE_log.hh ALE_mem.hh CoSieve.hh CoSifter.hh Delta.hh Filter.hh Map.hh \
	Mesh.hh ParDelta.hh Partitioner.hh Sieve.hh Sifter.hh
OBJSC    = ALE.o               
OBJSF    =
LIBBASE  = libpetscdm
MANSEC   = DA
LOCDIR   = src/dm/mesh/sieve/

ALE.o:  ALE.hh 
ALE.hh: ALE_mem.hh ALE_log.hh ALE_containers.hh 

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
