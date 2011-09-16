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

elliptic_solve: elliptic_solve.o tao_chkopts
	-${CLINKER} -o elliptic_solve elliptic_solve.o ${TAO_LIB} ${PETSC_LIB}
	${RM} elliptic_solve.o

elliptic_matmult: elliptic_matmult.o tao_chkopts
	-${CLINKER} -o elliptic_matmult elliptic_matmult.o ${TAO_LIB} ${PETSC_LIB}
	${RM} elliptic_matmult.o

mydot: mydot.o tao_chkopts
	-${CLINKER} -o mydot mydot.o ${TAO_LIB} ${PETSC_LIB}
	${RM} mydot.o

cavity: cavity.o tao_chkopts
	-${CLINKER} -o cavity cavity.o ${TAO_LIB} ${PETSC_LIB}
	${RM} cavity.o

elliptic_stage: elliptic_stage.o tao_chkopts
	-${CLINKER} -o elliptic_stage elliptic_stage.o ${TAO_LIB} ${PETSC_LIB}
	${RM} elliptic_stage.o
