# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: ex1

CFLAGS	        = $(CPPFLAGS) $(FCONF) -D__SDIR__='"$(LOCDIR)"' 
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

EXAMPLESC       = ex1.c ex2.c util2.c
EXAMPLESF       = ex1f.F ex2f.F ex2f.h

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) ex1.o

ex1f: ex1f.o  chkopts
	-$(FLINKER) -o ex1f ex1f.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex1f.o

ex2: ex2.o util2.o chkopts
	-$(CLINKER) -o ex2 ex2.o util2.o $(PETSC_LIB)
	$(RM) ex2.o util2.o

ex2f: ex2f.o util2.o chkopts
	-$(FLINKER) -o ex2f ex2f.o util2.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex2f.o util2.o

runex1:
	-@$(MPIRUN) -np 1 ex1  > ex1_1.tmp 2>&1;	  \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp

runex1f:
	-@$(MPIRUN) -np 1 ex1f  > ex1f_1.tmp 2>&1;	  \
	   if (diff output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   $(RM) -f ex1f_1.tmp

