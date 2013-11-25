ALL: 

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = eptorsion1.c eptorsion2.c minsurf2.c rosenbrock1.c
EXAMPLESF        = eptorsion2f.F rosenbrock1f.F
EXAMPLESCH       =
EXAMPLESFH       = eptorsion2f.h rosenbrock1f.h
TUTORIALS        = rosenbrock1.c eptorsion1.c minsurf1.c eptorsion2.c\
                   minsurf2.c eptorsion2f.F
LOCDIR		 = src/tao/unconstrained/examples/tutorials/
MANSEC		 = Unconstrained
DOCS		 = 
DIRS		 = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

rosenbrock1: rosenbrock1.o chkopts
	-${CLINKER} -o rosenbrock1 rosenbrock1.o ${PETSC_TAO_LIB}
	${RM} rosenbrock1.o

eptorsion2: eptorsion2.o chkopts
	-${CLINKER} -o eptorsion2 eptorsion2.o ${PETSC_TAO_LIB}
	${RM} eptorsion2.o

eptorsion1: eptorsion1.o chkopts
	-${CLINKER} -o eptorsion1 eptorsion1.o ${PETSC_TAO_LIB}
	${RM} eptorsion1.o

rosenbrock1f: rosenbrock1f.o chkopts
	-${FLINKER} -o rosenbrock1f rosenbrock1f.o ${PETSC_TAO_LIB} 
	${RM} rosenbrock1f.o

eptorsion2f: eptorsion2f.o chkopts
	-${FLINKER} -o eptorsion2f eptorsion2f.o ${PETSC_TAO_LIB}
	${RM} eptorsion2f.o

minsurf1: minsurf1.o chkopts
	-${CLINKER} -o minsurf1 minsurf1.o ${PETSC_TAO_LIB}
	${RM} minsurf1.o

minsurf2: minsurf2.o chkopts
	-${CLINKER} -o minsurf2 minsurf2.o ${PETSC_TAO_LIB}
	${RM} minsurf2.o


