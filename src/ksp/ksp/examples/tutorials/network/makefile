
LOCDIR		 = src/ksp/ksp/examples/tutorials/network
MANSEC           = KSP

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_KSP_LIB}
	${RM} ex1.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_KSP_LIB}
	${RM} ex2.o
#---------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ksp_monitor > ex1_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_1.tmp

runex1_2:
	-@${MPIEXEC} -n 2 ./ex1 -ksp_monitor > ex1_2.tmp 2>&1;   \
	   if (${DIFF} output/ex1_2.out ex1_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_2.tmp

runex2_1:
	-@${MPIEXEC} -n 1 ./ex2 -seed 0 -ksp_monitor > ex2_1.tmp 2>&1;   \
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_2.tmp

runex2_2:
	-@${MPIEXEC} -n 2 ./ex2 -seed 0 -ksp_monitor > ex2_2.tmp 2>&1;   \
	   if (${DIFF} output/ex2_2.out ex2_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex2_2.tmp
