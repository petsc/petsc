IPETSCDIR = ../../../
ALL: ex1

CFLAGS    = $(OPT) $(CONF) $(INCLUDEDIRS)
SOURCEC   = 
SOURCEF   =
WSOURCEC  = 
SOURCEH   =
OBJSC     =
WOBJS     = 
OBJSF     =
LIBBASE   = libpetscsnes
EXAMPLES  = ex1 ex2 ex3 ex4 ex5 ex6
EXAMPLESC = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c
LINCLUDE  = $(SOURCEH)


runexamples:
	-@echo "Examples in src/snes/examples -----------------"
	-@$(MPIRUN) ex1 -snes_monitor > ex1_1.tmp 2>&1;   \
           if (diff ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           $(RM) -f ex1_1.tmp
	-@$(MPIRUN) ex1 -snes_monitor -snes_method tr > ex1_2.tmp 2>&1;   \
           if (diff ex1_2.out ex1_2.tmp) then true; \
           else echo "Possible problem with ex1_2, diffs above"; fi; \
           $(RM) -f ex1_2.tmp
	-@$(MPIRUN) ex2 -snes_monitor > ex2_1.tmp 2>&1;   \
           if (diff ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi; \
           $(RM) -f ex2_1.tmp
	-@$(MPIRUN) ex2 -snes_monitor -snes_method tr > ex2_2.tmp 2>&1;   \
           if (diff ex2_2.out ex2_2.tmp) then true; \
           else echo "Possible problem with ex2_2, diffs above"; fi; \
           $(RM) -f ex2_2.tmp
	-@$(MPIRUN) ex3 -snes_monitor > ex3_1.tmp 2>&1;   \
           if (diff ex3_1.out ex3_1.tmp) then true; \
           else echo "Possible problem with ex3_1, diffs above"; fi; \
           $(RM) -f ex3_1.tmp
	-@$(MPIRUN) ex3 -snes_monitor -snes_method tr > ex3_2.tmp 2>&1;   \
           if (diff ex3_2.out ex3_2.tmp) then true; \
           else echo "Possible problem with ex3_2, diffs above"; fi; \
           $(RM) -f ex3_2.tmp
	-@$(MPIRUN) ex4 -snes_monitor > ex4_1.tmp 2>&1;   \
           if (diff ex4_1.out ex4_1.tmp) then true; \
           else echo "Possible problem with ex4_1, diffs above"; fi; \
           $(RM) -f ex4_1.tmp
	-@$(MPIRUN) ex4 -snes_monitor -snes_method tr > ex4_2.tmp 2>&1;   \
           if (diff ex4_2.out ex4_2.tmp) then true; \
           else echo "Possible problem with ex4_2, diffs above"; fi; \
           $(RM) -f ex1_2.tmp

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

ex6: ex6.o 
	-$(CLINKER) -o ex6 ex6.o $(PETSC_LIB)
	$(RM) ex6.o

ex7: ex7.o 
	-$(CLINKER) -o ex7 ex7.o $(PETSC_LIB)
	$(RM) ex7.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
