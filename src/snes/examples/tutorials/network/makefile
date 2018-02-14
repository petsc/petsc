#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

ALL: lib
CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = ex1.c
LOCDIR           = src/snes/examples/tutorials/network/
DIRS		 = pflow waternet
MANSEC           = SNES

OBJECTS_EX1 = ex1.o pflow/PFReadData.o pflow/pffunctions.o waternet/waternetreaddata.o waternet/waterfunctions.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: $(OBJECTS_EX1) chkopts
	-${CLINKER} -o ex1 $(OBJECTS_EX1) ${PETSC_SNES_LIB}
	${RM} ${OBJECTS_EX1}
#--------------------------------------------------------------------------

runex1:
	-@${MPIEXEC} -n 1 ./ex1 -coupled_snes_monitor_short -options_left no> ex1.tmp 2>&1;   \
	   if (${DIFF} output/ex1.out ex1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1.tmp
runex1_2:
	-@${MPIEXEC} -n 3 ./ex1 -coupled_snes_monitor_short -options_left no > ex1.tmp 2>&1;   \
	   if (${DIFF} output/ex1_2.out ex1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1.tmp

TESTEXAMPLES_C		       =  ex1.PETSc ex1.rm

include ${PETSC_DIR}/lib/petsc/conf/test
