IPETSCDIR = ../../../../
ALL: ex1

CFLAGS        = $(OPT) $(CONF) $(PETSC_INCLUDE)
SOURCEC       = 
SOURCEF       =
WSOURCEC      = 
SOURCEH       =
OBJSC         =
WOBJS         = 
OBJSF         =
LIBBASE       = libpetscsnes
EXAMPLES_1    = ex2
RUNEXAMPLES_1 = runex2 runex2_2 runex2_3
EXAMPLES_3    = ex1
RUNEXAMPLES_3 = runex1 runex1_2
EXAMPLESC     = ex1.c

FOBJ = deptfg.o dgl1fg.o dgl2fg.o dmsafg.o dodcfg.o dpjbfg.o dsscfg.o \
       depths.o dgl1hs.o dgl2hs.o dmsahs.o dodchs.o dpjbhs.o dsschs.o \
       dmsabc.o dodcps.o dgl2co.o mpinter.o

EX1 = ex1.o $(FOBJ)
EX4 = ex4.o $(FOBJ)

ex1: $(EX1) chkopts
	-$(CLINKER) -o ex1 $(EX1) $(PETSC_LIB)
	$(RM) $(EX1)

ex2: ex2.o chkopts
	-$(CLINKER) -o ex2 ex2.o $(PETSC_LIB)
	$(RM) ex2.o

ex3: ex3.o chkopts
	-$(CLINKER) -o ex3 ex3.o $(PETSC_LIB)
	$(RM) ex3.o

# These are just for some experiments
ex4: $(EX4) chkopts
	-$(CLINKER) -o ex4 $(EX4) $(PETSC_LIB)
	$(RM) ex4.o

ex5: ex5.o chkopts
	-$(CLINKER) -o ex5 ex5.o $(PETSC_LIB)
	$(RM) ex5.o

runex1:
	-@$(MPIRUN) ex1 -nox -snes_smonitor > ex1_1.tmp 2>&1;   \
           if (diff output/ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           $(RM) -f ex1_1.tmp
runex1_2:
	-@$(MPIRUN) ex1 -nox -snes_smonitor -snes_method umls > ex1_2.tmp 2>&1;\
           if (diff output/ex1_2.out ex1_2.tmp) then true; \
           else echo "Possible problem with ex1_2, diffs above"; fi; \
           $(RM) -f ex1_2.tmp
runex2:
	-@$(MPIRUN) ex2 -p 2 -snes_smonitor > ex2_1.tmp 2>&1;   \
           if (diff output/ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi; \
           $(RM) -f ex2_1.tmp
runex2_2:
	-@$(MPIRUN) ex2 -p 2 -snes_smonitor -snes_method umls > ex2_2.tmp 2>&1;\
           if (diff output/ex2_2.out ex2_2.tmp) then true; \
           else echo "Possible problem with ex2_2, diffs above"; fi; \
           $(RM) -f ex2_2.tmp
runex2_3:
	-@$(MPIRUN) ex2 -p 2 -snes_smonitor -snes_method umls -snes_mf > ex2_3.tmp 2>&1;\
           if (diff output/ex2_3.out ex2_3.tmp) then true; \
           else echo "Possible problem with ex2_3, diffs above"; fi; \
           $(RM) -f ex2_3.tmp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

