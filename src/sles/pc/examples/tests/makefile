IPETSCDIR = ../../../
ALL: ex1

CFLAGS   = $(OPT)  $(CONF) $(INCLUDEDIRS) $(BS_INCLUDE)

SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
EXAMPLES = ex1 ex2 ex3 ex4
REXAMPLES= runex1 runex2 runex3 runex4 runex4_2
CEXAMPLES= ex1 ex2 ex3 ex4 ex5 ex6
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c
LINCLUDE = $(SOURCEH)

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o $(SLES_LIB)
	$(RM) ex1.o

ex2: ex2.o 
	-$(CLINKER) -o ex2 ex2.o $(SLES_LIB)
	$(RM) ex2.o

ex3: ex3.o 
	-$(CLINKER) -o ex3 ex3.o $(SLES_LIB)
	$(RM) ex3.o

ex4: ex4.o 
	-$(CLINKER) -o ex4 ex4.o $(SLES_LIB)
	$(RM) ex4.o

ex5: ex5.o 
	-$(CLINKER) -o ex5 ex5.o $(SLES_LIB)
	$(RM) ex5.o

ex6: ex6.o 
	-$(CLINKER) -o ex6 ex6.o $(SLES_LIB)
	$(RM) ex6.o

runex1:
	-@$(MPIRUN) ex1 > ex1_1.tmp 2>&1;   \
           if (diff output/ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           $(RM) -f ex1_1.tmp 
runex2:
	-@$(MPIRUN) ex2 -ksp_method cg -ksp_smonitor > ex2_1.tmp 2>&1;   \
           if (diff output/ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi; \
           $(RM) -f ex2_1.tmp 
runex3:
	-@$(MPIRUN) ex3 -ksp_method gmres -ksp_monitor -pc_method sor\
                             -pc_sor_symmetric > ex3_1.tmp 2>&1;   \
           if (diff output/ex3_1.out ex3_1.tmp) then true; \
           else echo "Possible problem with ex3_1, diffs above"; fi; \
           $(RM) -f ex3_1.tmp 
runex4:
	-@$(MPIRUN) ex4 -ksp_method cg -ksp_monitor  > ex4_1.tmp 2>&1;   \
           if (diff output/ex4_1.out ex4_1.tmp) then true; \
           else echo "Possible problem with ex4_1, diffs above"; fi; \
           $(RM) -f ex4_1.tmp 
runex4_2:
	-@$(MPIRUN) ex4 -ksp_method bcgs -ksp_monitor  > ex4_2.tmp 2>&1;   \
           if (diff output/ex4_2.out ex4_2.tmp) then true; \
           else echo "Possible problem with ex4_2, diffs above"; fi; \
           $(RM) -f ex4_2.tmp 


include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
