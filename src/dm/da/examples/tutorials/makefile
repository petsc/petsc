
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        = 
LOCDIR          = src/dm/da/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c
EXAMPLESF       = 
MANSEC          = DA

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_DM_LIB}
	${RM} -f ex2.o

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_DM_LIB}
	${RM} -f ex3.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_DM_LIB}
	${RM} -f ex4.o

ex5: ex5.o   chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_DM_LIB}
	${RM} -f ex5.o

ex6: ex6.o   chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_DM_LIB}
	${RM} -f ex6.o
#-----------------------------------------------------------------
runex1:
	-@${MPIRUN} -np 2 ./ex1
runex2:
	-@${MPIRUN} -np 3 ./ex2
runex3:
	-@${MPIRUN} -np 3 ./ex3 -mat_view  > ex3_1.tmp 2>&1;   \
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIRUN} -np 3 ./ex3 -mat_view -dim 2 > ex3_2.tmp 2>&1;   \
	   if (${DIFF} output/ex3_2.out ex3_2.tmp) then true; \
	   else echo "Possible problem with ex3_2, diffs above"; fi; \
	   ${RM} -f ex3_2.tmp
runex3_3:
	-@${MPIRUN} -np 3 ./ex3 -mat_view -dim 3 > ex3_3.tmp 2>&1;   \
	   if (${DIFF} output/ex3_3.out ex3_3.tmp) then true; \
	   else echo "Possible problem with ex3_3, diffs above"; fi; \
	   ${RM} -f ex3_3.tmp
TESTEXAMPLES_C		  = ex3.PETSc runex3 runex3_2 runex3_3 ex3.rm ex4.PETSc ex4.rm ex6.PETSc ex6.rm
TESTEXAMPLES_C_X11	  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex5.PETSc ex5.rm
TESTEXAMPLES_FORTRAN	  = 
TESTEXAMPLES_C_X11_MPIUNI = 
TESTEXAMPLES_C_COMPLEX	  = 
TESTEXAMPLES_13		  = 

include ${PETSC_DIR}/bmake/common/test
