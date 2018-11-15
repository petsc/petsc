
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
CXXFLAGS	= -std=c++11 -I${ADOLC_BUILDDIR}/include
LOCDIR          = src/ts/examples/tutorials/autodiff/
EXAMPLESCXX     = ex16adj_ad.cpp ex16opt_ic_ad.cpp
EXAMPLESF       =
MANSEC          = TS
DIRS            =
CLEANFILES      = SA-data/*

LIB             = ${PETSC_TS_LIB} -L${USER_LIB} -lboost_system
LIB             += -L${ADOLC_BUILDDIR}/lib64 -ladolc -Wl,-rpath,${ADOLC_BUILDDIR}/lib64
LIB             += -L${COLPACK_HOME}/lib -lColPack -Wl,-rpath,${COLPACK_HOME}/lib -lm

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex16adj_ad: ex16adj_ad.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

ex16opt_ic_ad: ex16opt_ic_ad.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
