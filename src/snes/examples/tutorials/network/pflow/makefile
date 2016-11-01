#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

ALL:pf
CFLAGS	         = 
FFLAGS	         = 
CPPFLAGS         = 
FPPFLAGS         =
EXAMPLESC        = pf.c
LOCDIR           = src/snes/examples/tutorials/network/pflow/

OBJECTS_PF = pf.o PFReadData.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


pf: $(OBJECTS_PF) chkopts
	-${CLINKER} -o pf $(OBJECTS_PF) ${PETSC_SNES_LIB}
	${RM} $(OBJECTS_PF)

pf_jacerror: pf_jacerror.o PFReadData.o chkopts
	-${CLINKER} -o pf_jacerror pf_jacerror.o PFReadData.o ${PETSC_SNES_LIB}
	${RM} pf_jacerror.o PFReadData.o

#--------------------------------------------------------------------------
runpf:
	-@${MPIEXEC} -n 1 ./pf -snes_rtol 1.e-3 > pf.tmp 2>&1;   \
	   if (${DIFF} output/pf.out pf.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pf, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pf.tmp

runpf_2:
	-@${MPIEXEC} -n 3 ./pf -snes_rtol 1.e-3 > pf.tmp 2>&1;   \
	   if (${DIFF} output/pf.out pf.tmp) then true; \
	   else printf "${PWD}\nPossible problem with pf_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f pf.tmp

runpf_jacerror:
	-@${MPIEXEC} -n 3 ./pf_jacerror -snes_type test -snes_test_display_threshold 10.0


TESTEXAMPLES_C		       =  pf.PETSc runpf runpf_2 pf.rm

include ${PETSC_DIR}/lib/petsc/conf/test
