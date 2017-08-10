#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

ALL: lib
CFLAGS	         = -I/scratch/hzhang/wash/src/
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = ex1.c ex2.c
LOCDIR           = src/snes/examples/tutorials/network/
DIRS		 = pflow waternet
MANSEC           = SNES

OBJECTS_EX1 = pflow/PFReadData.o waternet/waternetreaddata.o waternet/waterfunctions.o

WASH_LOC = /scratch/hzhang/wash
OBJECTS_EX2 = pflow/PFReadData.o ${WASH_LOC}/src/washnetwork.o ${WASH_LOC}/src/parse.o ${WASH_LOC}/src/pipe/pipeImpls.o ${WASH_LOC}/src/pipe/pipeInterface.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o $(OBJECTS_EX2) chkopts
	-${CLINKER} -o ex1 ex1.o $(OBJECTS_EX1) ${PETSC_SNES_LIB}
	${RM} ex1.o

ex2: ex2.o $(OBJECTS_EX2) chkopts
	-${CLINKER} -o ex2 ex2.o $(OBJECTS_EX2) ${PETSC_SNES_LIB}
	${RM} ex2.o


#--------------------------------------------------------------------------


runex2:
	-@${MPIEXEC} -n 1 ./ex2 -snes_rtol 1.e-3 > ex2.tmp 2>&1;   \
	   if (${DIFF} output/ex2.out ex2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex2.tmp
runex2_2:
	-@${MPIEXEC} -n 3 ./ex2 -snes_rtol 1.e-3 -petscpartitioner_type simple > ex2.tmp 2>&1;   \
	   if (${DIFF} output/ex2_2.out ex2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex2_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex2.tmp
runex2_epa:
	-@${MPIEXEC} -n 1 ./ex2 -washdata ${WASH_LOC}/tests/cases/HG-SP-1-1.inp -ksp_monitor -pc_type lu -pc_factor_shift_type NONZERO
runex2_epa_2:
	-@${MPIEXEC} -n 4 ./ex2 -washdata ${WASH_LOC}/tests/cases/HG-SP-1-1.inp -ksp_monitor -pc_type lu

TESTEXAMPLES_C		       =  ex1.PETSc ex1.rm 

include ${PETSC_DIR}/lib/petsc/conf/test
