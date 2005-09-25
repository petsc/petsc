
SIDL        = ex1.sidl
SIDLEXCLUDE = --exclude-external
TLIBNAME    = libex1
REMOVEFILES = ./server/c++/TOPS* ./server/c++/sidl* ./server/c++/gov_cca*
DIRS        = server/c++
EXNUM       = 1

ex1:ex1.cc
	        ${CC_LINKER}  -o ex1 -g ${MPI_include} -I../../../server/c++/ -I../../../client/c++ ${BABEL_INCLUDE}  ex1.cc -L${PETSC_LIB_DIR} -lex1 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -lclient-c++ ${PETSC_LIB} ${BABEL_LIB} ${SYS_lib}

include ${PETSC_DIR}/src/tops/makefile.rules

