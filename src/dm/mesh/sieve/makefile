ALL: lib

CFLAGS   = -I. ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE}
FFLAGS   =
CPPFLAGS = -DALE_USE_DEBUGGING -DALE_HAVE_CXX_ABI -DALE_USE_LOGGING -DALE_LOGGING_USE_EVENTS -DALE_LOGGING_LOG_MEM -DALE_LOGGING_VERBOSE #-DALE_LOGGING_USE_STAGES
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
