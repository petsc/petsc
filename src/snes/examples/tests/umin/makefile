PETSC_DIR = ../../../..
ALL: ex1

/* Note that we need CONF to pick up the Fortran format info */
CFLAGS        = $(CONF) $(PCONF) $(PETSC_INCLUDE)
FFLAGS        = 
SOURCEC       = 
SOURCEF       =
SOURCEH       =
OBJSC         =
OBJSF         =
LIBBASE       = libpetscsnes
EXAMPLES_1    = ex2 ex3
RUNEXAMPLES_1 = runex2 runex2_2 runex3
EXAMPLES_2    = 
RUNEXAMPLES_2 = 
EXAMPLES_3    = ex1
RUNEXAMPLES_3 = runex1 runex1_2
EXAMPLESC     = ex1.c ex2.c ex3.c
DIRS          = 

FOBJ = deptfg.o dgl1fg.o dgl2fg.o dmsafg.o dodcfg.o dpjbfg.o dsscfg.o \
       depths.o dgl1hs.o dgl2hs.o dmsahs.o dodchs.o dpjbhs.o dsschs.o \
       dmsabc.o dodcps.o dgl2co.o mpinter.o

ex1: ex1.o $(FOBJ) chkopts
	-$(CLINKER) -o ex1 ex1.o $(FOBJ)  $(PETSC_LIB)
	$(RM) ex1.o $(FOBJ)

ex2: ex2.o chkopts
	-$(CLINKER) -o ex2 ex2.o $(PETSC_LIB)
	$(RM) ex2.o

ex3: ex3.o chkopts
	-$(CLINKER) -o ex3 ex3.o $(PETSC_LIB)
	$(RM) ex3.o

runex1:
	-@$(MPIRUN) ex1 -nox -snes_smonitor > ex1_1.tmp 2>&1;   \
           if (diff output/ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           $(RM) -f ex1_1.tmp
runex1_2:
	-@$(MPIRUN) ex1 -nox -snes_smonitor -snes_type umls > ex1_2.tmp 2>&1;\
           if (diff output/ex1_2.out ex1_2.tmp) then true; \
           else echo "Possible problem with ex1_2, diffs above"; fi; \
           $(RM) -f ex1_2.tmp
runex2:
	-@$(MPIRUN) ex2 -p 2 -snes_smonitor > ex2_1.tmp 2>&1;   \
           if (diff output/ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi; \
           $(RM) -f ex2_1.tmp
runex2_2:
	-@$(MPIRUN) ex2 -p 2 -snes_smonitor -snes_type umls > ex2_2.tmp 2>&1;\
           if (diff output/ex2_2.out ex2_2.tmp) then true; \
           else echo "Possible problem with ex2_2, diffs above"; fi; \
           $(RM) -f ex2_2.tmp
runex2_3:
	-@$(MPIRUN) ex2 -p 2 -snes_smonitor -snes_type umls -snes_mf > ex2_3.tmp 2>&1;\
           if (diff output/ex2_3.out ex2_3.tmp) then true; \
           else echo "Possible problem with ex2_3, diffs above"; fi; \
           $(RM) -f ex2_3.tmp
runex3:
	-@$(MPIRUN) -np 2 ex3 -snes_smonitor > ex3_1.tmp 2>&1;\
           if (diff output/ex3_1.out ex3_1.tmp) then true; \
           else echo "Possible problem with ex3_1, diffs above"; fi;\
	   $(RM) -f ex3_1.tmp

runex3_2:
	-@$(MPIRUN) -np 2 ex3 -Ny 2 -snes_type umls -snes_smonitor > ex3_2.tmp 2>&1;\
           if (diff output/ex3_2.out ex3_2.tmp) then true; \
           else echo "Possible problem with ex3_2, diffs above"; fi;\
	   $(RM) -f ex3_2.tmp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
