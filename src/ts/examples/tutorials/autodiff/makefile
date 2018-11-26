
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
CXXFLAGS	=
LOCDIR          = src/ts/examples/tutorials/autodiff/
EXAMPLESCXX     = adr_ex1.cpp adr_ex5adj.cpp adr_ex5adj_mf ex16adj.cpp ex16opt_ic.cpp
EXAMPLESF       =
MANSEC          = TS
DIRS            =
CLEANFILES      = SA-data/* *.tap

LIB             = ${PETSC_TS_LIB}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

adr_ex1: adr_ex1.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

adr_ex5adj: adr_ex5adj.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

adr_ex5adj_mf: adr_ex5adj_mf.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

ex16adj: ex16adj.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

ex16opt_ic: ex16opt_ic`.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
