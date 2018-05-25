
# This directory contains access to matlab test problems for the derivative free
# least squares solvers.

ALL:

CFLAGS           =
FFLAGS           =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/tao/leastsquares/examples/matlab/
MANSEC           = TAO
EXAMPLESC        = taopounders.c
EXAMPLESF        = 
EXAMPLESCH       =
EXAMPLESFH       = 
EXAMPLESMATLAB   = TestingInitialize.m TestingFinalize.m ProblemInitialize.m ProblemFinalize.m
DIRS             = 

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

taopounders: taopounders.o chkopts
	-${CLINKER} -o taopounders taopounders.o ${PETSC_TAO_LIB}
	${RM} taopounders.o

include ${PETSC_DIR}/lib/petsc/conf/test

