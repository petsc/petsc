ITOOLSDIR = ../../../
ALL: ex1

CFLAGS   = $(OPT)  $(CONF) $(INCLUDEDIRS)
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
EXAMPLES = ex1 ex2 ex3 ex4 ex5
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c ex5.c
SYSLIB   = $(SLES_LIB) -lm
LINCLUDE = $(SOURCEH)

runexamples: 
	-@echo "Examples in src/pc/examples -----------------"
	-@$(MPIRUN) ex1 > ex1_1.tmp 2>&1;   \
           if (diff ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           $(RM) -f ex1_1.tmp 
	-@$(MPIRUN) ex2 -kspmethod cg -kspmonitor > ex2_1.tmp 2>&1;   \
           if (diff ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi; \
           $(RM) -f ex2_1.tmp 
	-@$(MPIRUN) ex3 -kspmethod gmres -kspmonitor -pcmethod sor\
                             -sor_symmetric > ex3_1.tmp 2>&1;   \
           if (diff ex3_1.out ex3_1.tmp) then true; \
           else echo "Possible problem with ex3_1, diffs above"; fi; \
           $(RM) -f ex3_1.tmp 
	-@$(MPIRUN) ex4 -kspmethod cg -kspmonitor  > ex4_1.tmp 2>&1;   \
           if (diff ex4_1.out ex4_1.tmp) then true; \
           else echo "Possible problem with ex4_1, diffs above"; fi; \
           $(RM) -f ex4_1.tmp 
	-@$(MPIRUN) ex4 -kspmethod bcgs -kspmonitor  > ex4_2.tmp 2>&1;   \
           if (diff ex4_2.out ex4_2.tmp) then true; \
           else echo "Possible problem with ex4_2, diffs above"; fi; \
           $(RM) -f ex4_2.tmp 

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o $(LIBNAME) $(SYSLIB)
	$(RM) ex1.o

ex2: ex2.o 
	-$(CLINKER) -o ex2 ex2.o $(LIBNAME) $(SYSLIB)
	$(RM) ex2.o

ex3: ex3.o 
	-$(CLINKER) -o ex3 ex3.o $(LIBNAME) $(SYSLIB)
	$(RM) ex3.o

ex4: ex4.o 
	-$(CLINKER) -o ex4 ex4.o $(LIBNAME) $(SYSLIB)
	$(RM) ex4.o

ex5: ex5.o 
	-$(CLINKER) -o ex5 ex5.o $(LIBNAME) $(SYSLIB)
	$(RM) ex5.o

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
