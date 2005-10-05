
EXNAME      = Ex4
SIDL        = ${EXNAME}.sidl
TLIBNAME    = lib${EXNAME}
SIDLEXCLUDE = -e TOPS[\.a-zA-Z_]* -e gov.cca[\.a-zA-Z_]*
TOPSCLIENT_LIB = -L${PETSC_LIB_DIR} ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -ltopsclient-c++
DIRS        = server/c++

ex4:ex4.cc 
	${CC_LINKER} -o ex4 -g ${MPI_include} -I../../../server/c++/ -I../../../client/c++ ${BABEL_INCLUDE}  ex4.cc -L${PETSC_LIB_DIR} -lex4 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -lclient-c++ ${PETSC_LIB} ${BABEL_LIB} ${SYS_lib}

include ${PETSC_DIR}/src/tops/makefile.rules
