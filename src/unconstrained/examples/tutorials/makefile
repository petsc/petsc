ALL: 

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/unconstrained/examples/tutorials/
MANSEC		 = Unconstrained
DOCS		 = 
DIRS		 = 

include ${TAO_DIR}/conf/tao_base

rosenbrock1: rosenbrock1.o tao_chkopts
	-${CLINKER} -o rosenbrock1 rosenbrock1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} rosenbrock1.o

eptorsion2: eptorsion2.o tao_chkopts
	-${CLINKER} -o eptorsion2 eptorsion2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} eptorsion2.o

eptorsion1: eptorsion1.o tao_chkopts
	-${CLINKER} -o eptorsion1 eptorsion1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} eptorsion1.o

rosenbrock1f: rosenbrock1f.o tao_chkopts
	-${FLINKER} -o rosenbrock1f rosenbrock1f.o ${TAO_LIB} ${PETSC_LIB} 
	${RM} rosenbrock1f.o

eptorsion2f: eptorsion2f.o tao_chkopts
	-${FLINKER} -o eptorsion2f eptorsion2f.o ${TAO_LIB} ${PETSC_LIB}
	${RM} eptorsion2f.o



