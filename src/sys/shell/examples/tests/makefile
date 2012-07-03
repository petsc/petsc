ALL: ${TESTS} 

CFLAGS      = 
FFLAGS      = 
CPPFLAGS    = ${EXTRA_FLAGS} 
SOURCEC     = testcomponentsI.c testcomponentsII.c
SOURCEF     = 
SOURCEH     =
OBJSC       = testcomponentsI.o testcomponentsII.o
OBJSF       = 
MANSEC      = Sys
LOCDIR      = src/sys/shell/examples/tests/


include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
LIBBASE     = libtestcomponents
LIBNAME    = ${INSTALL_LIB_DIR}/${LIBBASE}.${AR_LIB_SUFFIX}
SHLIBS = ${LIBBASE}
OTHERSHAREDLIBS=${PETSC_LIB}

TESTS = test1

libs: chklib_dir lib shared

test1: test1.c
	make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} TEST=test1 test

TEST       = test1
test: libs ${TEST}.o
	${SL_LINKER} -o ${TEST} ${TEST}.o ${PETSC_LIB};\
	${RM} ${EX}.o 



distclean: clean
	${RM} *~ *.o ${TESTS} 
	${RM} -rf ${TESTLIBDIR}

.PHONY: distclean

TESTEXAMPLES_C_X	  = 
TESTEXAMPLES_C_X_MPIUNI = 



