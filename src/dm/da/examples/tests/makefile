# $Id: makefile,v 1.65 2001/08/22 18:02:41 balay Exp $ 

CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/da/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c\
                  ex11.c ex12.c ex12.m ex13.c ex14.c ex15.c ex16.c ex17.c ex18.c ex19.c
EXAMPLESF       = 
MANSEC          = DA

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts 
	-${CLINKER} -o ex2 ex2.o  ${PETSC_DM_LIB}
	${RM} -f ex2.o

ex3: ex3.o  chkopts 
	-${CLINKER} -o ex3 ex3.o  ${PETSC_DM_LIB}
	${RM} -f ex3.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_DM_LIB}
	${RM} -f ex4.o

ex5: ex5.o  chkopts 
	-${CLINKER} -o ex5 ex5.o  ${PETSC_DM_LIB}
	${RM} -f ex5.o

ex6: ex6.o   chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_DM_LIB}
	${RM} -f ex6.o

ex7: ex7.o   chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_DM_LIB}
	${RM} -f ex7.o

ex8: ex8.o   chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_DM_LIB}
	${RM} -f ex8.o

ex9: ex9.o   chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_DM_LIB}
	${RM} -f ex9.o

ex10: ex10.o   chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_DM_LIB}
	${RM} -f ex10.o

ex11: ex11.o   chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_DM_LIB}
	${RM} -f ex11.o

ex12: ex12.o   chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_DM_LIB}
	${RM} -f ex12.o

ex13: ex13.o   chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_DM_LIB}
	${RM} -f ex13.o

ex14: ex14.o   chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_DM_LIB}
	${RM} -f ex14.o

ex15: ex15.o   chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_DM_LIB}
	${RM} -f ex15.o

ex16: ex16.o   chkopts
	-${CLINKER} -o ex16 ex16.o  ${PETSC_DM_LIB}
	${RM} -f ex16.o

ex17: ex17.o   chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_DM_LIB}
	${RM} -f ex17.o

ex18: ex18.o   chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_DM_LIB}
	${RM} -f ex18.o

ex19: ex19.o   chkopts
	-${CLINKER} -o ex19 ex19.o  ${PETSC_DM_LIB}
	${RM} -f ex19.o
#-------------------------------------------------------------------------------
runex1:
	-@${MPIRUN} -np 2 ex1 -nox > ex1_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp
runex2:
	-@${MPIRUN} -np 2 ex2 > ex2_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp
runex3:
	-@${MPIRUN} -np 3 ex3 -time 50
runex4:
	-@${MPIRUN} -np 4 ex4  -nox > ex4_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   ${RM} -f ex4_1.tmp
runex5:
	-@${MPIRUN} -np 3 ex5 -time 50
runex6:
	-@${MPIRUN} -np 6 ex6 -test_order
runex12:
	-@echo "path([getenv('PETSC_DIR') '/bin/matlab'],path); ex12(2) ; exit" | ${MATLABCOMMAND} -display ${DISPLAY} > ex12_1.tmp 2>&1; grep -v flops ex12_1.tmp | grep -v Completed | grep -v About | grep -v Color | grep -v "at line 27" > ex12_1.tmp2 ; if (${DIFF} output/ex12_1.out ex12_1.tmp2) then true; \
	   else echo "Possible problem with PETSC SOCKET MATLAB INTERFACE ex12_1, diffs above"; fi; \
	   ${RM} -f ex12_1.tmp ex12_1.tmp2

# ex13 uses the output of ex14
runex13:
	-@${MPIRUN} -np 3 ex13
	-@${RM} -f daoutput daoutput.info 
runex14:
	-@${MPIRUN} -np 4 ex14 ; sleep 1

TESTEXAMPLES_1  = ex1.PETSc runex1 ex1.rm ex4.PETSc runex4 ex4.rm
TESTEXAMPLES_2  = ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm ex5.PETSc\
		  runex5 ex5.rm ex6.PETSc runex6 ex6.rm ex7.PETSc ex7.rm \
                  ex14.PETSc runex14 ex14.rm ex13.PETSc runex13 ex13.rm
TESTEXAMPLES_3  =
TESTEXAMPLES_4  = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc\
		  ex3.rm ex4.PETSc ex4.rm ex5.PETSc ex5.rm\
		  ex6.PETSc ex6.rm ex12.PETSc ex12.rm
TESTEXAMPLES_5  =
TESTEXAMPLES_13 = ex8.PETSc ex8.rm ex9.PETSc ex9.rm ex10.PETSc ex10.rm \
		  ex11.PETSc ex11.rm
TESTEXAMPLES_MATLAB = ex12.PETSc runex12 ex12.rm

include ${PETSC_DIR}/bmake/common/test
