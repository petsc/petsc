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
EXAMPLES_2    = ex1
RUNEXAMPLES_2 = runex1 runex1_2
EXAMPLESC     = ex1.c

FOBJ = deptfg.o dgl1fg.o dgl2fg.o dmsafg.o dodcfg.o dpjbfg.o dsscfg.o \
       depths.o dgl1hs.o dgl2hs.o dmsahs.o dodchs.o dpjbhs.o dsschs.o \
       dmsabc.o dodcps.o dgl2co.o mpinter.o

EX1 = ex1.o $(FOBJ)

ex1: $(EX1) chkopts
	-$(CLINKER) -o ex1 $(EX1) $(PETSC_LIB)
	$(RM) ex1.o

runex1:
	-@$(MPIRUN) ex1 -nox -snes_smonitor > ex1_1.tmp 2>&1;   \
           if (diff output/ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           $(RM) -f ex1_1.tmp
runex1_2:
	-@$(MPIRUN) ex1 -nox -snes_smonitor -umls > ex1_2.tmp 2>&1;   \
           if (diff output/ex1_2.out ex1_2.tmp) then true; \
           else echo "Possible problem with ex1_2, diffs above"; fi; \
           $(RM) -f ex1_2.tmp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
