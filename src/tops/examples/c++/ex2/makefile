
SIDL        = ex2.sidl
SIDLEXCLUDE = -e=TOPS[a-zA-Z_]*
TLIBNAME    = libex2
REMOVEFILES = ./server/c++/TOPS* ./server/c++/sidl* ./server/c++/gov_cca*
DIRS        = server/c++

ex2:ex2.cc 
	${CC_LINKER} -o ex2 -g ${MPI_include} -I../../../server/c++/ -I../../../client/c++ -I${BABEL_include}  ex2.cc -L${PETSC_LIB_DIR} -lex2 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -lclient-c++ ${PETSC_LIB} -L${BABEL_lib} -lsidl ${SYS_lib}

include ${PETSC_DIR}/src/tops/makefile.rules
