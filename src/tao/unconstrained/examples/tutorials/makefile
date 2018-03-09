ALL:

CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = eptorsion1.c eptorsion2.c minsurf2.c rosenbrock1.c spectraladjointassimilation.c
EXAMPLESF        = eptorsion2f.F rosenbrock1f.F90
EXAMPLESCH       =
EXAMPLESFH       = eptorsion2f.h rosenbrock1f.h
LOCDIR		 = src/tao/unconstrained/examples/tutorials/
MANSEC		 =
DOCS		 =
DIRS		 =
CLEANFILES       = eptorsion1 eptorsion2 minsurf2 rosenbrock1 eptorsion2f rosenbrock1f

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

spectraladjointassimilation: spectraladjointassimilation.o chkopts
	-${CLINKER} -o spectraladjointassimilation spectraladjointassimilation.o ${PETSC_TAO_LIB}  ${PETSC_TS_LIB}
	${RM} spectraladjointassimilation.o

burgers_spectral: burgers_spectral.o chkopts
	-${CLINKER} -o burgers_spectral burgers_spectral.o ${PETSC_TAO_LIB}  ${PETSC_TS_LIB}
	${RM} burgers_spectral.o

heat-data-assimulation: heat-data-assimulation.o chkopts
	-${CLINKER} -o heat-data-assimulation heat-data-assimulation.o ${PETSC_TAO_LIB}  ${PETSC_TS_LIB}
	${RM} heat-data-assimulation.o

rosenbrock1: rosenbrock1.o chkopts
	-${CLINKER} -o rosenbrock1 rosenbrock1.o ${PETSC_TAO_LIB}
	${RM} rosenbrock1.o
	
rosenbrock2: rosenbrock2.o chkopts
	-${CLINKER} -o rosenbrock2 rosenbrock2.o ${PETSC_TAO_LIB}
	${RM} rosenbrock2.o

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

include ${PETSC_DIR}/lib/petsc/conf/test
