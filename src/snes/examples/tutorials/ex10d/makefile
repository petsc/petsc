
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/snes/examples/tutorials/ex10d/
EXAMPLESC	 = ex10.c
EXAMPLESF	 =
MANSEC          = SNES

include ${PETSC_DIR}/bmake/common/base

ex10: ex10.o  chkopts
	-$(CLINKER) -o ex10 ex10.o $(PETSC_LIB)
	$(RM) ex10.o
#-----------------------------------------------------------------
runex10:
	-@$(MPIRUN) -np 2 ./ex10 -snes_smonitor > ex10_1.tmp 2>&1;   \
	   if (${DIFF} output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Possible problem with ex10_1, diffs above"; fi; \
	   ${RM} -f ex10_1.tmp

TESTEXAMPLES_6	 = ex10.PETSc runex10 ex10.rm

include ${PETSC_DIR}/bmake/common/test



