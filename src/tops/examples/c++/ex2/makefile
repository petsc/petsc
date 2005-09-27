
SIDL        = ex2.sidl
SIDLEXCLUDE = -e TOPS[\.a-zA-Z_]* -e gov.cca[\.a-zA-Z_]*
TLIBNAME    = libex2
DIRS        = server/c++
EXNUM       = 2

ex2:ex2.cc 
	${CC_LINKER} -o ex2 -g ${MPI_include} -I../../../server/c++/ -I../../../client/c++ ${BABEL_INCLUDE} ex2.cc -L${PETSC_LIB_DIR} -lex2 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -lclient-c++ ${PETSC_LIB} ${BABEL_LIB} ${SYS_lib}

include ${PETSC_DIR}/src/tops/makefile.rules
