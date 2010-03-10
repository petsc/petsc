ALL: ${TESTS} 

CFLAGS      = 
FFLAGS      = 
CPPFLAGS    = ${EXTRA_FLAGS} 
SOURCEC     = testcomponentsI.c testcomponentsII.c
SOURCEF     = 
SOURCEH     =
OBJSC       = testcomponentsI.o testcomponentsII.o
OBJSF       = 
MANSEC      = fwk
LOCDIR      = src/sys/fwk/examples/tests/


include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
LIBBASE     = libtestcomponents
LIBNAME    = ${INSTALL_LIB_DIR}/${LIBBASE}.${AR_LIB_SUFFIX}
SHLIBS = ${LIBBASE}
OTHERSHAREDLIBS=${PETSC_LIB_BASIC}

TESTS = test1

libs: chklib_dir lib shared

test1:
	make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} EX=test1 ex

EX_DIR     = .
TESTS       = test1
ex: libs chkex_dir ${EX}.o
	${SL_LINKER} -o ${EX} ${EX}.o ${PETSC_LIB_BASIC};\
	${MV} ${EX} ${EX_DIR}/;\
	${RM} ${EX}.o 

chkex_dir:
	@if [ ! -d "${EX_DIR}" ]; then \
	  echo Making directory ${EX_DIR} for executables; \
	  ${MKDIR} ${EX_DIR}; fi


distclean: clean
	${RM} *~ *.o ${TESTS} 
	${RM} -rf ${TESTLIBDIR}

.PHONY: distclean

TESTEXAMPLES_C_X11	  = 
TESTEXAMPLES_C_X11_MPIUNI = 



