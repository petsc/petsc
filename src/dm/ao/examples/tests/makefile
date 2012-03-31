
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =	 
LOCDIR          = src/dm/ao/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex4.c ex7.c
EXAMPLESF       = ex4f.F
MANSEC          = AO

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts 
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_DM_LIB}
	${RM} -f ex2.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_DM_LIB}
	${RM} -f ex4.o
ex4f: ex4f.o  chkopts
	-${FLINKER} -o ex4f ex4f.o ${PETSC_DM_LIB}
	${RM} -f ex4f.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_DM_LIB}
	${RM} -f ex7.o
#-------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 > ex1_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex1_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC}  -n 2 ./ex1 > ex1_2.tmp 2>&1;\
	   if (${DIFF} output/ex1_2.out ex1_2.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex1_2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1_2.tmp
runex1_3:
	-@${MPIEXEC}  -n 3 ./ex1 > ex1_3.tmp 2>&1;\
	   if (${DIFF} output/ex1_3.out ex1_3.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex1_3, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1_3.tmp
runex1_4:
	-@${MPIEXEC}  -n 3 ./ex1 -ao_type basic > ex1_3.tmp 2>&1;\
	   if (${DIFF} output/ex1_3.out ex1_3.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex1_4, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1_3.tmp

runex2:
	-@${MPIEXEC} -n 1  ./ex2 > ex2_1.tmp 2>&1;\
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex2_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex2_1.tmp
runex2_2:
	-@${MPIEXEC}  -n 2 ./ex2 > ex2_2.tmp 2>&1;\
	   if (${DIFF} output/ex2_2.out ex2_2.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex2_2, diffs above \n========================================="; fi;\
	   ${RM} -f ex2_2.tmp
runex2_3:
	-@${MPIEXEC} -n 3 ./ex2 > ex2_3.tmp 2>&1;\
	   if (${DIFF} output/ex2_3.out ex2_3.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex2_3, diffs above \n========================================="; fi;\
	   ${RM} -f ex2_3.tmp

runex4:
	-@${MPIEXEC}  -n 1 ./ex4 > ex4.tmp 2>&1;\
	   if (${DIFF} output/ex4.out ex4.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex4.tmp
runex4_2:
	-@${MPIEXEC}  -n 2 ./ex4 > ex4.tmp 2>&1;\
	   if (${DIFF} output/ex4.out ex4.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4_2, diffs above \n========================================="; fi;\
	   ${RM} -f ex4.tmp

runex4f:
	-@${MPIEXEC}  -n 1 ./ex4f > ex4.tmp 2>&1;\
	   if (${DIFF} output/ex4.out ex4.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4f_1, diffs above \n========================================="; fi;\
	   ${RM} -f ex4.tmp
runex4f_2:
	-@${MPIEXEC}  -n 2 ./ex4f > ex4.tmp 2>&1;\
	   if (${DIFF} output/ex4.out ex4.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4f_2, diffs above \n========================================="; fi;\
	   ${RM} -f ex4.tmp

runex7:
	-@${MPIEXEC}  -n 2 ./ex7 > ex7_1.tmp 2>&1;\
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex7_1, diffs above \n========================================="; fi ;\
	   ${RM} -f ex7_1.tmp
TESTEXAMPLES_C		  = ex1.PETSc runex1 runex1_2 runex1_3 ex1.rm ex2.PETSc runex2 runex2_2 runex2_3 ex2.rm \
                            ex4.PETSc runex4 runex4_2 ex4.rm  ex7.PETSc runex7 ex7.rm
TESTEXAMPLES_C_X	  = 
TESTEXAMPLES_FORTRAN	  = ex4f.PETSc runex4f runex4f_2 ex4f.rm
TESTEXAMPLES_C_X_MPIUNI = 
TESTEXAMPLES_C3		  = 

include ${PETSC_DIR}/conf/test
