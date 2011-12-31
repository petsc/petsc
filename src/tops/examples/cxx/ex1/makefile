
EXNAME      = Ex1
SIDL        = ${EXNAME}.sidl
TLIBNAME    = lib${EXNAME}
SIDLEXCLUDE = -e TOPS[\.a-zA-Z_]* -e gov.cca[\._a-zA-Z_]*
TOPSCLIENT_LIB = -L${PETSC_LIB_DIR}/tops/cxx ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR}/tops/cxx -ltopsclient-cxx 
DIRS        = server/cxx

ex1:ex1.cxx
	        ${PCC_LINKER}  -o ex1 -g ${MPI_include} -I../../../server/cxx -I../../../client/cxx ${BABEL_INCLUDE}  ex1.cxx -L${PETSC_LIB_DIR} -lex1 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -lclient-cxx ${PETSC_LIB} ${BABEL_LIB} ${SYS_lib}

include ${PETSC_DIR}/src/tops/makefile.rules

