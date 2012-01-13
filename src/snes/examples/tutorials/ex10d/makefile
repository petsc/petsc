
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/snes/examples/tutorials/ex10d/
EXAMPLESC	 = ex10.c
EXAMPLESF	 =
MANSEC          = SNES

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex10: ex10.o  chkopts
	-$(CLINKER) -o ex10 ex10.o $(PETSC_LIB)
	$(RM) ex10.o
#-----------------------------------------------------------------
runex10:
	-@$(MPIEXEC) -n 2 ./ex10 -snes_monitor_short > ex10_1.tmp 2>&1;   \
	   if (${DIFF} output/ex10_1.out ex10_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex10_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex10_1.tmp

runex10_2:
	-@$(MPIEXEC) -n 2 ./ex10 -snes_monitor_short -fd_jacobian_coloring > ex10_2.tmp 2>&1;   \
	   if (${DIFF} output/ex10_2.out ex10_2.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex10_2, diffs above \n========================================="; fi; \
	   ${RM} -f ex10_2.tmp

TESTEXAMPLES_C_NOCOMPLEX	 = ex10.PETSc runex10 runex10_2 ex10.rm

include ${PETSC_DIR}/conf/test



