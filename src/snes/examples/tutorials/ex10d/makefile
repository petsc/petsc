
CFLAGS		 =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/snes/examples/tutorials/ex10d/
EXAMPLESC	 = ex10.c
EXAMPLESF	 =
MANSEC          = SNES

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex10: ex10.o  chkopts
	-$(CLINKER) -o ex10 ex10.o $(PETSC_LIB)
	$(RM) ex10.o

include ${PETSC_DIR}/lib/petsc/conf/test



