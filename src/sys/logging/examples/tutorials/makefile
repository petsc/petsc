
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/logging/examples/tutorials/
EXAMPLESC       =
EXAMPLESF       = ex1f.F90
MANSEC          = Sys
SUBMANSEC       = Profiling

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_LIB}
	${RM} -f ex1f.o

runex1f:
	-@${MPIEXEC} -n 2 ./ex1f -log_view ascii:filename.xml:ascii_xml  > ex1_1.tmp 2>&1; \
	if (${DIFF} output/ex1f.out ex1_1.tmp) then true; \
	else printf "${PWD}\nPossible problem with ex1_1, diffs above\n========================= \n"; fi; \
	${RM} -f ex1_1.tmp filename.xml

TESTEXAMPLES_FORTRAN	=  ex1f.PETSc runex1f ex1f.rm

include ${PETSC_DIR}/lib/petsc/conf/test
