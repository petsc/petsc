
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/logging/examples/tutorials/
EXAMPLESC       =
EXAMPLESF       =
MANSEC          = Profiling

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_LIB}
	${RM} -f ex1f.o

runex1f:
	-@${MPIEXEC} -n 2 ./ex1f -log_view ascii:filename.xml:ascii_xml

TESTEXAMPLES_FORTRAN	=  ex1f.PETSc runex1f ex1f.rm

include ${PETSC_DIR}/lib/petsc/conf/test
