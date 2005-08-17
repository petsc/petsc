
SIDL        = ex1.sidl
SIDLEXCLUDE = -e=TOPS[a-zA-Z_]*
TLIBNAME    = libex1
REMOVEFILES = ./server/c++/TOPS* ./server/c++/sidl*

ex1:ex1.cc 
	${CC_LINKER} -o ex1 -g ${MPI_include} -I../../../server/c++/ -I../../../client/c++ -I${BABEL_include}  ex1.cc -L${PETSC_LIB_DIR} -lex1 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -lclient-c++ ${PETSC_LIB} -L${BABEL_lib} -lsidl ${SYS_lib}

include ${PETSC_DIR}/src/tops/makefile.rules
