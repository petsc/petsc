include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules.doc

ex13: ex13.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}
#	${CLINKER} -g $@ $^ ${PETSC_LIB}

