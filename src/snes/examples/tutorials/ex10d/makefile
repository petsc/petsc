# $Id: makefile,v 1.2 1999/03/19 17:51:25 kaushik Exp balay $ 

ALL: ex10

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/snes/examples/tutorials/ex10
TESTEXAMPLES_1	 = ex10.PETSc runex10 ex10.rm
BUILDEXAMPLES_1	 = ex10.PETSc ex10.rm
DOCS		 = readme
DIRS		 = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex10: ex10.o  chkopts
	-$(CLINKER) -o ex10 ex10.o $(PETSC_LIB)
	$(RM) ex10.o

runex10:
	-@$(MPIRUN) -np 2 ex10 -snes_monitor > ex10_1.tmp 2>&1;   \
	   if (diff output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Possible problem with ex10_1, diffs above"; fi; \
	   ${RM} -f ex10_1.tmp

