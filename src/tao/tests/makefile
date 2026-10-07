-include ../../../petscdir.mk

CLEANFILES  = taoboundtest taotermtest1 taotermtest2 taosnes cutest

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

cutest: cutest.o
	${CLINKER} -o $@ $< ${CUTEST_LIB} ${PETSC_TAO_LIB}

cutest.o: ${PETSC_DIR}/include/petsc/private/taocutestimpl.h
