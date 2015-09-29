ALL:WASH
CFLAGS	         = 
FFLAGS	         = 
CPPFLAGS         = 
FPPFLAGS         =
EXAMPLESC        = pipes1.c
LOCDIR           = src/ts/examples/tutorials/network/wash/

OBJECTS_PIPE = pipeInterface.o pipeImpls.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

VPATH = pipe
pipes1: pipes1.o $(OBJECTS_PIPE) chkopts
	-${CLINKER} -o pipes1 pipes1.o $(OBJECTS_PIPE) ${PETSC_DM_LIB}
	${RM} pipes1.o $(OBJECTS_PIPE)

#--------------------------------------
runpipes1:
	-@${MPIEXEC} -n 1 ./pipes1 -ts_monitor -snes_converged_reason -ksp_converged_reason -case 1 -ts_max_steps 1 -nox > pipes1.tmp 2>&1;\
	   if (${DIFF} output/pipes1_1.out pipes1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pipes1_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pipes1.tmp
runpipes1_2:
	-@${MPIEXEC} -n 2 ./pipes1 -ts_monitor -snes_converged_reason -ksp_converged_reason -case 1 -ts_max_steps 1 -nox > pipes1.tmp 2>&1;\
	   if (${DIFF} output/pipes1_2.out pipes1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pipes1_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pipes1.tmp
runpipes1_3:
	-@${MPIEXEC} -n 4 ./pipes1 -ts_monitor -snes_converged_reason -ksp_converged_reason -case 1 -ts_max_steps 1 -nox > pipes1.tmp 2>&1;\
	   if (${DIFF} output/pipes1_3.out pipes1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pipes1_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pipes1.tmp

TESTEXAMPLES_C		       = pipes1.PETSc runpipes1 runpipes1_2 runpipes1_3 pipes1.rm