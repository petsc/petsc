-include ../../../../petscdir.mk
CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
MANSEC           = Vec
LOCDIR		 = src/vec/vec/tutorials/
EXAMPLESC	 = ex1.c ex2.c ex3.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c \
                   ex11.c ex12.c ex15.c ex16.c ex18.c ex19.c ex42.c ex42a.c ex21.c ex43.c performance.c
EXAMPLESF	 = ex1f.F90 ex1f90.F90 ex2f.F ex3f.F ex4f.F ex4f90.F90 ex5f.F90 ex6f.F90 ex9f.F90 \
                   ex11f.F ex11f90.F90 ex12f.F90 ex14f.F90 ex16f.F90 ex20f90.F90 ex21f90.F90 ex43f.F90

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex7: ex7.o ex7f.o
ex21f90: ex21f90.o ex21.o
ex43f: ex43.o ex43f.o

testex31: ex31.PETSc
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/vec/vec/tutorials/ex31 with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/vec/vec/tutorials/ex31 manually"; \
	elif [ -f ex31 ]; then \
           ${MPIEXEC} -n 1 ./ex31 > ex31_1.tmp 2>&1; \
	   if (${DIFF} output/ex31_1.testout ex31_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/vec/vec/tutorials/ex31 run successfully with Matlab engine"; \
	   else echo "Possible error running C/C++ src/vec/vec/tutorials/ex31 with Matlab engine"; \
           echo "See http://www.mcs.anl.gov/petsc/documentation/faq.html";\
           cat ex31_1.tmp; fi;\
        fi; \

