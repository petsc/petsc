# $Id: makefile,v 1.8 1997/09/10 18:54:40 balay Exp balay $ 

ALL: ex1

CFLAGS	        = $(CPPFLAGS) $(FCONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscec
LOCDIR          = src/ec/examples/tests/
TESTEXAMPLES_1  = 
TESTEXAMPLES_2  = 
TESTEXAMPLES_4  = 
BUILDEXAMPLES_1 = 
BUILDEXAMPLES_2 = 
BUILDEXAMPLES_4 = 
EXAMPLESC       = 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o $(PETSC_LIB)
	$(RM) ex1.o

ex2: ex2.o 
	-$(CLINKER) -o ex2 ex2.o $(PETSC_LIB)
	$(RM) ex2.o

ex3: ex3.o 
	-$(CLINKER) -o ex3 ex3.o $(PETSC_LIB)
	$(RM) ex3.o

ex4: ex4.o 
	-$(CLINKER) -o ex4 ex4.o $(PETSC_LIB)
	$(RM) ex4.o

ex5: ex5.o 
	-$(CLINKER) -o ex5 ex5.o $(PETSC_LIB)
	$(RM) ex5.o

runex1:
	-@$(MPIRUN) -np 1 ex1 > ex1_1.tmp 2>&1;   \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp 

