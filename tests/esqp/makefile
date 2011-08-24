LOCDIR=tests/
EXAMPLESC = test_link1.c linesearch1.c

include ${TAO_DIR}/conf/tao_base

loadq: loadq.o tao_chkopts
	-${CLINKER} -o loadq loadq.o ${TAO_LIB} ${PETSC_LIB}
	${RM} loadq.o

elliptic: elliptic.o tao_chkopts
	-${CLINKER} -o elliptic elliptic.o ${TAO_LIB} ${PETSC_LIB}
	${RM} elliptic.o

parabolic: parabolic.o tao_chkopts
	-${CLINKER} -o parabolic parabolic.o ${TAO_LIB} ${PETSC_LIB}
	${RM} parabolic.o

hyperbolic: hyperbolic.o tao_chkopts
	-${CLINKER} -o hyperbolic hyperbolic.o ${TAO_LIB} ${PETSC_LIB}
	${RM} hyperbolic.o

testksp: testksp.o tao_chkopts
	-${CLINKER} -o testksp testksp.o ${TAO_LIB} ${PETSC_LIB}
	${RM} testksp.o

testksp2: testksp2.o tao_chkopts
	-${CLINKER} -o testksp2 testksp2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} testksp2.o
