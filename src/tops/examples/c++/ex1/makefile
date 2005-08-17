
SIDL        = ex1.sidl
SIDLEXCLUDE = -e=TOPS[a-zA-Z_]*
TLIBNAME    = libex1
REMOVEFILES = ./server/c++/TOPS* ./server/c++/sidl*

ex1:ex1.cc 
	${MPI_HOME}/bin/mpicxx -o ex1 -g ${MPI_include} -I../../../server/c++/ -I../../../client/c++ -I${BABEL_include}  ex1.cc -L${PETSC_LIB_DIR} -lex1 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -lclient-c++ -lpetscsnes -L${BABEL_lib} -lsidl ${SYS_lib} -lpetscdm -lpetscksp -lpetscmat -lpetscvec -lpetsc -L${MPI_HOME}/lib -lfmpich -lpmpich -lmpichfsup
	#${CC_LINKER} -o ex1 -g ${MPI_include} -I../../../server/c++/ -I../../../client/c++ -I${BABEL_include}  ex1.cc -L${PETSC_LIB_DIR} -lex1 -ltops ${CC_LINKER_SLFLAG}${PETSC_LIB_DIR} -lclient-c++ -lpetscsnes -L${BABEL_lib} -lsidl ${SYS_lib} -lpetscdm -lpetscksp -lpetscmat -lpetscvec -lpetsc

include ${PETSC_DIR}/src/tops/makefile.rules
