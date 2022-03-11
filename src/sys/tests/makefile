-include ../../../petscdir.mk
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c ex11.c ex12.c \
                  ex14.c ex15.c ex16.c ex18.c ex19.c ex20.c ex21.c \
                  ex22.c ex23.c ex24.c ex25.c ex26.c ex27.c ex28.c ex29.c ex30.c ex31.c ex32.c ex33.c ex35.c ex37.c \
                  ex44.cxx ex45.cxx ex46.cxx ex47.c ex49.c \
                  ex50.c ex51.c ex52.c ex53.c ex54.c ex56.c ex57.c ex58.c
EXAMPLESF       = ex1f.F90 ex5f.F ex6f.F ex17f.F ex36f.F90 ex38f.F90 ex47f.F90 ex48f90.F90 ex49f.F90
MANSEC          = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

testex55:
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/sys/tests/ex55.py with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/sys/tests/ex55.py manually"; \
	else \
           export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}${PETSC_MPI4PY_PYTHONPATH}; \
           echo "Using PYTHONPATH=$${PYTHONPATH} to locate mpi4py for test"; \
           ${MPIEXEC} -n 2 ${PYTHON} ./ex55.py > ex55_1.tmp  2>&1;\
	   if (${DIFF} output/ex55_1.testout ex55_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ mpi4py python example src/sys/tests/ex55.py run successfully with 2 MPI process"; \
	   else echo "Possible error running Python src/sys/tests/ex55.py with 2 MPI processes"; \
           echo "See http://www.mcs.anl.gov/petsc/documentation/faq.html";\
           cat ex55_1.tmp; fi; \
        ${RM} -f ex55.tmp; fi
