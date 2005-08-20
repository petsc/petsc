
SIDL        = ex3.sidl
SIDLEXCLUDE = -e=TOPS[a-zA-Z_]*
TLIBNAME    = libex3
REMOVEFILES = ./server/c++/TOPS* ./server/c++/sidl* ./server/c++/gov_cca*
DIRS        = server/c++

ex3:ex3.cc 
	${CC_LINKER} -o ex3 -g ${MPI_include} -I../../../server/c++/ -I../../../client/c++ -I${BABEL_include}  ex3.cc -L${PETSC_LIB_DIR} -lex3 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -lclient-c++ ${PETSC_LIB} -L${BABEL_lib} -lsidl ${SYS_lib}

include ${PETSC_DIR}/src/tops/makefile.rules
