
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =	 
LOCDIR          = src/dm/ao/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c
EXAMPLESF       = 
MANSEC          = AO

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o  chkopts 
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_DM_LIB}
	${RM} -f ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_DM_LIB}
	${RM} -f ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_DM_LIB}
	${RM} -f ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_DM_LIB}
	${RM} -f ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_DM_LIB}
	${RM} -f ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_DM_LIB}
	${RM} -f ex7.o
#-------------------------------------------------------------------
runex1:
	-@${MPIRUN} -np 1 ./ex1 > ex1_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIRUN}  -np 2 ./ex1 > ex1_2.tmp 2>&1;\
	   if (${DIFF} output/ex1_2.out ex1_2.tmp) then true ;  \
	   else echo "Possible problem with ex1_2, diffs above"; fi ;\
	   ${RM} -f ex1_2.tmp
runex1_3:
	-@${MPIRUN}  -np 3 ./ex1 > ex1_3.tmp 2>&1;\
	   if (${DIFF} output/ex1_3.out ex1_3.tmp) then true ;  \
	   else echo "Possible problem with ex1_3, diffs above"; fi ;\
	   ${RM} -f ex1_3.tmp
runex2:
	-@${MPIRUN} -np 1  ./ex2 > ex2_1.tmp 2>&1;\
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi;\
	   ${RM} -f ex2_1.tmp
runex2_2:
	-@${MPIRUN}  -np 2 ./ex2 > ex2_2.tmp 2>&1;\
	   if (${DIFF} output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi;\
	   ${RM} -f ex2_2.tmp
runex2_3:
	-@${MPIRUN} -np 3 ./ex2 > ex2_3.tmp 2>&1;\
	   if (${DIFF} output/ex2_3.out ex2_3.tmp) then true; \
	   else echo "Possible problem with ex2_3, diffs above"; fi;\
	   ${RM} -f ex2_3.tmp

runex3:
	-@${RM} -f ex3_1.tmp dataoutput dataoutput.info
	-@${MPIRUN} -np 2 ./ex3 > ex3_1.tmp 2>&1;   \
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi
	-@${RM} -f ex3_1.tmp dataoutput dataoutput.info

runex5:
	-@${MPIRUN} -np 2 ./ex5 > ex5_1.tmp 2>&1;   \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp

runex7:
	-@${MPIRUN}  -np 2 ./ex7 > ex7_1.tmp 2>&1;\
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true ;  \
	   else echo "Possible problem with ex7_1, diffs above"; fi ;\
	   ${RM} -f ex7_1.tmp
TESTEXAMPLES_C		  = ex1.PETSc runex1 runex1_2 runex1_3 ex1.rm ex2.PETSc runex2 runex2_2 runex2_3 ex2.rm \
                            ex3.PETSc runex3 ex3.rm ex5.PETSc runex5 ex5.rm ex7.PETSc runex7 ex7.rm
TESTEXAMPLES_C_X11	  = 
TESTEXAMPLES_FORTRAN	  = 
TESTEXAMPLES_C_X11_MPIUNI = 
TESTEXAMPLES_C3		  = ex4.PETSc ex4.rm ex6.PETSc ex6.rm

include ${PETSC_DIR}/bmake/common/test
