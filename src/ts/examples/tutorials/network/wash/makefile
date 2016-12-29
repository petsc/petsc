ALL:WASH
CFLAGS	         = 
FFLAGS	         = 
CPPFLAGS         = 
FPPFLAGS         =
EXAMPLESC        = pipes1.c pipeInterface.c pipeImpls.c
LOCDIR           = src/ts/examples/tutorials/network/wash/

OBJECTS_PIPE = pipeInterface.o pipeImpls.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

pipes1: pipes1.o $(OBJECTS_PIPE) chkopts
	-${CLINKER} -o pipes1 pipes1.o $(OBJECTS_PIPE) ${PETSC_TS_LIB}
	${RM} pipes1.o $(OBJECTS_PIPE)

#--------------------------------------
runpipes1:
	-@${MPIEXEC} -n 1 ./pipes1 -ts_monitor -case 1 -ts_max_steps 1 -nox > pipes1.tmp 2>&1;\
	   if (${DIFF} output/pipes1_1.out pipes1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pipes1_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pipes1.tmp
# parallel runs with '--with-64-bit-indices' fails, likely caused by partitioner packages. 
runpipes1_2:
	-@${MPIEXEC} -n 2 ./pipes1 -ts_monitor -case 1 -ts_max_steps 1 -petscpartitioner_type simple -nox > pipes1.tmp 2>&1;\
	   if (${DIFF} output/pipes1_1.out pipes1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pipes1_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pipes1.tmp
runpipes1_3:
	-@${MPIEXEC} -n 4 ./pipes1 -ts_monitor -case 1 -ts_max_steps 1 -petscpartitioner_type simple -nox > pipes1.tmp 2>&1;\
	   if (${DIFF} output/pipes1_1.out pipes1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pipes1_3, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pipes1.tmp

runpipes1_4:
	-@${MPIEXEC} -n 1 ./pipes1 -ts_monitor -case 0 -ts_max_steps 1 -nox > pipes1.tmp 2>&1;\
	   if (${DIFF} output/pipes1_4.out pipes1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pipes1_4, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pipes1.tmp
runpipes1_5:
	-@${MPIEXEC} -n 2 ./pipes1 -ts_monitor -case 0 -ts_max_steps 1 -petscpartitioner_type simple -nox > pipes1.tmp 2>&1;\
	   if (${DIFF} output/pipes1_4.out pipes1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pipes1_5, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pipes1.tmp
runpipes1_6:
	-@${MPIEXEC} -n 4 ./pipes1 -ts_monitor -case 0 -ts_max_steps 1 -nox -petscpartitioner_type simple > pipes1.tmp 2>&1;\
	   if (${DIFF} output/pipes1_4.out pipes1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pipes1_6, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pipes1.tmp

runpipes1_7:
	-@${MPIEXEC} -n 1 ./pipes1 -ts_monitor -case 2 -ts_max_steps 1 -nox > pipes1.tmp 2>&1;\
	   if (${DIFF} output/pipes1_7.out pipes1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pipes1_7, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pipes1.tmp
runpipes1_8:
	-@${MPIEXEC} -n 2 ./pipes1 -ts_monitor -case 2 -ts_max_steps 1 -petscpartitioner_type simple -nox > pipes1.tmp 2>&1;\
	   if (${DIFF} output/pipes1_7.out pipes1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pipes1_8, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pipes1.tmp
runpipes1_9:
	-@${MPIEXEC} -n 4 ./pipes1 -ts_monitor -case 2 -ts_max_steps 1 -petscpartitioner_type simple -nox > pipes1.tmp 2>&1;\
	   if (${DIFF} output/pipes1_7.out pipes1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pipes1_9, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pipes1.tmp

TESTEXAMPLES_C		       = pipes1.PETSc runpipes1 runpipes1_2 runpipes1_3 runpipes1_4  \
                                 runpipes1_5 runpipes1_6 runpipes1_7 runpipes1_8 runpipes1_9 pipes1.rm

include ${PETSC_DIR}/lib/petsc/conf/test
