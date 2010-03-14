
CFLAGS	        = 
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ksp/pc/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c
EXAMPLESF       =
MANSEC          = PC

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_KSP_LIB} 
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_KSP_LIB} 
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER}  -o ex3 ex3.o ${PETSC_KSP_LIB} 
	${RM} ex3.o
#------------------------------------------------------------------------------------
runex3:
	-@${MPIEXEC} -n 1 ./ex3 -ksp_monitor_short > ex3_inode.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex3 -ksp_monitor_short -mat_no_inode > ex3.tmp 2>&1; \
	if (${DIFF} ex3_inode.out ex3.tmp) then true; \
	   else echo "Possible problem with ex3, diffs above"; fi; \
	   ${RM} -f ex3_inode.out; ${RM} -f ex3.tmp

runex3_pd:
	-@${MPIEXEC} -n 1 ./ex3 -ksp_monitor_short -pc_factor_shift_type POSITIVE_DEFINITE > ex3_inode.out 2>&1;
	-@${MPIEXEC} -n 1 ./ex3 -ksp_monitor_short -pc_factor_shift_type POSITIVE_DEFINITE -mat_no_inode > ex3.tmp 2>&1; \
	if (${DIFF} ex3_inode.out ex3.tmp) then true; \
	   else echo "Possible problem with ex3, diffs above"; fi; \
	   ${RM} -f ex3_inode.out; ${RM} -f ex3.tmp

TESTEXAMPLES_C	   = ex3.PETSc runex3 runex3_pd ex3.rm 
TESTEXAMPLES_C_X11 = ex2.PETSc ex2.rm

include ${PETSC_DIR}/conf/test

