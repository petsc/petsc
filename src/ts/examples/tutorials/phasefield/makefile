
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/phasefield
EXAMPLESC       = potentials.c heat.c biharmonic.c biharmonic2.c biharmonic3.c convert.c
EXAMPLESF       = 
EXAMPLESFH      = 
MANSEC          = TS

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

convert: convert.o  chkopts
	-${CLINKER} -o convert convert.o  ${PETSC_TS_LIB}
	${RM} convert.o

potentials: potentials.o  chkopts
	-${CLINKER} -o potentials potentials.o  ${PETSC_TS_LIB}
	${RM} potentials.o

heat: heat.o  chkopts
	-${CLINKER} -o heat heat.o  ${PETSC_TS_LIB}
	${RM} heat.o

biharmonic: biharmonic.o  chkopts
	-${CLINKER} -o biharmonic biharmonic.o  ${PETSC_TS_LIB}
	${RM} biharmonic.o

biharmonic2: biharmonic2.o  chkopts
	-${CLINKER} -o biharmonic2 biharmonic2.o  ${PETSC_TS_LIB}
	${RM} biharmonic2.o

biharmonic3: biharmonic3.o  chkopts
	-${CLINKER} -o biharmonic3 biharmonic3.o  ${PETSC_TS_LIB}
	${RM} biharmonic3.o


TESTEXAMPLES_C		  = potentials.PETSc potentials.rm  heat.PETSc heat.rm biharmonic.PETSc biharmonic.rm biharmonic2.PETSc biharmonic2.rm
TESTEXAMPLES_C_X	  = 

include ${PETSC_DIR}/conf/test
