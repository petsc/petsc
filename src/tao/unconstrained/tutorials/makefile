-include ../../../../petscdir.mk

CLEANFILES  = eptorsion1 eptorsion2 minsurf2 rosenbrock1 eptorsion2f rosenbrock1f rosenbrock2 rosenbrock3 cutest

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

rosenbrock4.o: rosenbrock4.h

rosenbrock4cu.o: rosenbrock4.h

cutest: cutest.o
	${CLINKER} -o $@ $< ${CUTEST_LIB} ${PETSC_TAO_LIB}

cutest.o: ${PETSC_DIR}/include/petsc/private/taocutestimpl.h
