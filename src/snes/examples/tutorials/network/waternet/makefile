#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

ALL:waternet
CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = waternet.c
LOCDIR           = src/snes/examples/tutorials/network/waternet/

OBJECTS_WATERNET = waternetreaddata.o waterfunctions.o waternet.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


waternet: $(OBJECTS_WATERNET) chkopts
	-${CLINKER} -o waternet $(OBJECTS_WATERNET) ${PETSC_SNES_LIB}
	${RM} $(OBJECTS_WATERNET)

#--------------------------------------------------------------------------
runwaternet:
	-@${MPIEXEC} -n 1 ./waternet -waternet_snes_converged_reason > waternet.tmp 2>&1;   \
	   if (${DIFF} output/waternet.out waternet.tmp) then true; \
	   else printf "${PWD}\nPossible problem with waternet, diffs above\n=========================================\n"; fi; \
	   ${RM} -f waternet.tmp

runwaternet_2:
	-@${MPIEXEC} -n 3 ./waternet -waternet_snes_converged_reason > waternet.tmp 2>&1;   \
	   if (${DIFF} output/waternet.out waternet.tmp) then true; \
	   else printf "${PWD}\nPossible problem with waternet_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f waternet.tmp

TESTEXAMPLES_C		       =  waternet.PETSc runwaternet runwaternet_2 waternet.rm

include ${PETSC_DIR}/lib/petsc/conf/test
