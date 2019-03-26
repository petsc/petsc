
# This directory contains access to matlab test problems for the derivative free
# least squares solvers.

ALL:

CFLAGS           =
FFLAGS           =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/tao/leastsquares/examples/matlab/
MANSEC           = Tao
EXAMPLESC        = matlab_ls_test.c
EXAMPLESF        = 
EXAMPLESCH       =
EXAMPLESFH       = 
EXAMPLESMATLAB   = TestingInitialize.m TestingFinalize.m ProblemInitialize.m ProblemFinalize.m
DIRS             = 

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

matlab_ls_test: matlab_ls_test.o chkopts
	-${CLINKER} -o matlab_ls_test matlab_ls_test.o ${PETSC_TAO_LIB}
	${RM} matlab_ls_test.o

include ${PETSC_DIR}/lib/petsc/conf/test

