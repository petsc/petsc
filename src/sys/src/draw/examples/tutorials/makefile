#PETSC_DIR = ../../../..
ALL: ex1

CFLAGS	        = $(CPPFLAGS) -D__DIR__='"$(LOCDIR)"'
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscsys
LOCDIR          = src/draw/examples/tutorials/
TESTEXAMPLES_2  = 
TESTEXAMPLES_3  =
TESTEXAMPLES_4  =
BUILDEXAMPLES_2 =
BUILDEXAMPLES_3 =
BUILDEXAMPLES_4 =
EXAMPLESC       =
EXAMPLESF       =

runex1:
	-@$(MPIRUN) ex1
runex2:
	-@$(MPIRUN) ex2
runex3:
	-@$(MPIRUN) ex3
runex4:
	-@$(MPIRUN) ex4
runex4_2:
	-@$(MPIRUN) -np 2 ex4

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) -f ex1.o

ex2: ex2.o  chkopts
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) -f ex2.o

ex3: ex3.o  chkopts
	-$(CLINKER) -o ex3 ex3.o  $(PETSC_LIB)
	$(RM) -f ex3.o

ex4: ex4.o  chkopts
	-$(CLINKER) -o ex4 ex4.o  $(PETSC_LIB)
	$(RM) -f ex4.o

ex5: ex5.o  chkopts
	-$(FLINKER) -o ex5 ex5.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) -f ex5.o

ex6: ex6.o  chkopts
	-$(CLINKER) -o ex6 ex6.o  $(PETSC_LIB)
	$(RM) -f ex6.o

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
