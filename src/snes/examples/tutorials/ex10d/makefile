# $Id: makefile,v 1.1 1999/03/19 04:47:03 bsmith Exp kaushik $ 

ALL: ex10

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/snes/examples/tutorials/ex10
DOCS		 = readme
DIRS		 = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex10: ex10.o  chkopts
	-$(CLINKER) -o ex10 ex10.o $(PETSC_LIB)
	$(RM) ex10.o

runex10_2:
	-@$(MPIRUN) -np 2 ex10 -ksp_monitor -snes_monitor
