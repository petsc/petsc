#PETSC_DIR = ../../../..
ALL: ex1

CFLAGS	        = $(CPPFLAGS) -D__DIR__='"$(LOCDIR)"' 
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscts
LOCDIR          = src/ts/examples/tutorials/
TESTEXAMPLES_1  = ex1.PETSc runex1 ex1.rm
TESTEXAMPLES_2  = 
TESTEXAMPLES_3  = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_4  =  
BUILDEXAMPLES_1 =
BUILDEXAMPLES_3 =

EXAMPLESC       = ex1.c
EXAMPLESF       = ex1f.F
#

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) ex1.o

ex1f: ex1f.o  chkopts
	-$(FLINKER) -o ex1f ex1f.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex1f.o

runex1:
	-@$(MPIRUN) ex1  > ex1_1.tmp 2>&1;	  \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp

runex1f:
	-@$(MPIRUN) ex1f  > ex1f_1.tmp 2>&1;	  \
	   if (diff output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   $(RM) -f ex1f_1.tmp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
