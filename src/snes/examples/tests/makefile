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
EXAMPLES  = ex3 ex4 ex5 ex6
REXAMPLES = runex3 runex3_2 runex4 runex4_2 runex5 runex5_2 runex6
CEXAMPLES = ex1 ex2 ex3 ex4 ex5 ex6 ex7
EXAMPLESC = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c
LINCLUDE  = $(SOURCEH)

runex3:
	-@$(MPIRUN) ex3 -nox -snes_smonitor > ex3_1.tmp 2>&1;   \
           if (diff output/ex3_1.out ex3_1.tmp) then true; \
           else echo "Possible problem with ex3_1, diffs above"; fi; \
           $(RM) -f ex3_1.tmp
runex3_2:
	-@$(MPIRUN) ex3 -nox -snes_smonitor -snes_method tr > ex3_2.tmp 2>&1; \
           if (diff output/ex3_2.out ex3_2.tmp) then true; \
           else echo "Possible problem with ex3_2, diffs above"; fi; \
           $(RM) -f ex3_2.tmp
runex4:
	-@$(MPIRUN) ex4 -snes_smonitor -nox > ex4_1.tmp 2>&1;   \
           if (diff output/ex4_1.out ex4_1.tmp) then true; \
           else echo "Possible problem with ex4_1, diffs above"; fi; \
           $(RM) -f ex4_1.tmp
runex4_2:
	-@$(MPIRUN) ex4 -snes_smonitor -nox -snes_method tr > ex4_2.tmp 2>&1;\
           if (diff output/ex4_2.out ex4_2.tmp) then true; \
           else echo "Possible problem with ex4_2, diffs above"; fi; \
           $(RM) -f ex4_2.tmp
runex5:
	-@$(MPIRUN) ex5 -snes_smonitor > ex5_1.tmp 2>&1;   \
           if (diff output/ex5_1.out ex5_1.tmp) then true; \
           else echo "Possible problem with ex5_1, diffs above"; fi; \
           $(RM) -f ex5_1.tmp
runex5_2:
	-@$(MPIRUN) ex5 -snes_mf > ex5_2.tmp 2>&1;   \
           if (diff output/ex5_2.out ex5_2.tmp) then true; \
           else echo "Possible problem with ex5_2, diffs above"; fi; \
           $(RM) -f ex5_2.tmp
runex6:
	-@$(MPIRUN) -np 4 ex6   > ex6_1.tmp 2>&1;   \
           if (diff output/ex6_1.out ex6_1.tmp) then true; \
           else echo "Possible problem with ex6_1, diffs above"; fi; \
           $(RM) -f ex6_1.tmp

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o $(PETSC_LIB)
	$(RM) ex1.o

ex2: ex2.o  chkopts
	-$(CLINKER) -o ex2 ex2.o $(PETSC_LIB)
	$(RM) ex2.o

ex3: ex3.o  chkopts
	-$(CLINKER) -o ex3 ex3.o $(PETSC_LIB)
	$(RM) ex3.o

ex4: ex4.o  chkopts
	-$(CLINKER) -o ex4 ex4.o $(PETSC_LIB)
	$(RM) ex4.o

ex5: ex5.o  chkopts
	-$(CLINKER) -o ex5 ex5.o $(PETSC_LIB)
	$(RM) ex5.o

ex6: ex6.o  chkopts
	-$(CLINKER) -o ex6 ex6.o $(PETSC_LIB)
	$(RM) ex6.o

ex7: ex7.o  chkopts
	-$(CLINKER) -o ex7 ex7.o $(PETSC_LIB)
	$(RM) ex7.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
