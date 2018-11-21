
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
CXXFLAGS	=
LOCDIR          = src/ts/examples/tutorials/autodiff/
EXAMPLESCXX     = ex16adj_ad.cpp ex16opt_ic_ad.cpp
EXAMPLESF       =
MANSEC          = TS
DIRS            =
CLEANFILES      = SA-data/*

LIB             = ${PETSC_TS_LIB}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex16adj_ad: ex16adj_ad.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

ex16opt_ic_ad: ex16opt_ic_ad.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
