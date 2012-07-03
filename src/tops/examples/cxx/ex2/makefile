
EXNAME      = Ex2
SIDL        = ${EXNAME}.sidl
TLIBNAME    = lib${EXNAME}
SIDLEXCLUDE = -e TOPS[\.a-zA-Z_]* -e gov.cca[\.a-zA-Z_]*
TOPSCLIENT_LIB = -L${PETSC_LIB_DIR} ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -ltops -L${PETSC_LIB_DIR}/tops/cxx ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR}/tops/cxx -ltopsclient-cxx 
DIRS        = server/cxx

ex2:ex2.cxx
	${PCC_LINKER} -o ex2 -g ${MPI_include} -I../../../server/cxx -I../../../client/cxx ${BABEL_INCLUDE} ex2.cxx -L${PETSC_LIB_DIR} -lex2 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -lclient-c++ ${PETSC_LIB} ${BABEL_LIB} ${SYS_lib}

include ${PETSC_DIR}/src/tops/makefile.rules
