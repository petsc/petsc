#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

ALL:pf
CFLAGS	         = -I/scratch/hzhang/wash/src/
FFLAGS	         = 
CPPFLAGS         = 
FPPFLAGS         =
EXAMPLESC        = pfc.c
LOCDIR           = src/snes/examples/tutorials/network/pflow/

OBJECTS_PF2 = pf2.o PFReadData.o
WASH_LOC = /scratch/hzhang/wash/src/
OBJECTS_EX1 = PFReadData.o ${WASH_LOC}/washnetwork.o ${WASH_LOC}/parse.o ${WASH_LOC}/pipe/pipeImpls.o ${WASH_LOC}/pipe/pipeInterface.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


pfc: pfc.o PFReadData.o chkopts
	-${CLINKER} -o pfc pfc.o PFReadData.o ${PETSC_SNES_LIB}
	${RM} pfc.o PFReadData.o

pf2: $(OBJECTS_PF2) chkopts
	-${CLINKER} -o pf2 $(OBJECTS_PF2) ${PETSC_SNES_LIB}
	${RM} $(OBJECTS_PF2)

ex1: ex1.o $(OBJECTS_EX1) chkopts
	-${CLINKER} -o ex1 ex1.o $(OBJECTS_EX1) ${PETSC_SNES_LIB}
	${RM} ex1.o

runex1:
	-@${MPIEXEC} -n 1 ./ex1 -snes_rtol 1.e-3 > ex1.tmp 2>&1;   \
	   if (${DIFF} output/ex1.out ex1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1.tmp
runex1_2:
	-@${MPIEXEC} -n 3 ./ex1 -snes_rtol 1.e-3 -petscpartitioner_type simple > ex1.tmp 2>&1;   \
	   if (${DIFF} output/ex1_2.out ex1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1.tmp
include ${PETSC_DIR}/lib/petsc/conf/test
