ITOOLSDIR = ../../../
ALL: ex1

CFLAGS    = $(OPT) $(CONF) $(INCLUDEDIRS)
SOURCEC   = 
SOURCEF   =
WSOURCEC  = 
SOURCEH   =
OBJSC     =
WOBJS     = 
OBJSF     =
LIBBASE   = libpetscsles
EXAMPLES  = ex1 ex2 ex3 ex4 ex5 ex6 ex7
EXAMPLESC = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c
SYSLIB    = -L$(LDIR) -lpetscstencil -lpetscgrid \
            -lpetscsles -lpetscksp -lpetscmat \
           -lpetscvec -lpetscsys -lpetscdraw \
           $(LAPACK_LIB) $(BLAS_LIB) $(BS_LIB) $(X11_LIB) \
           $(MPI_LIB) $(SYS_LIB) $(FC_LIB) -lm 
LINCLUDE  = $(SOURCEH)

runexamples:
	-@echo "Examples in src/sles/examples -----------------"
	-@$(MPIRUN) ex1 -kspmonitor > ex1_1.tmp 2>&1;   \
           if (diff ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           $(RM) -f ex1_1.tmp
	-@$(MPIRUN) -np 2 ex1  -kspmonitor > ex1_2.tmp 2>&1;   \
           if (diff ex1_2.out ex1_2.tmp) then true; \
           else echo "Possible problem with ex1_2, diffs above"; fi; \
           $(RM) -f ex1_2.tmp
	-@$(MPIRUN) -np 1 ex1 -pcmethod sor -sor_symmetric -kspmonitor >\
           ex1_3.tmp 2>&1;   \
           if (diff ex1_3.out ex1_3.tmp) then true; \
           else echo "Possible problem with ex1_3, diffs above"; fi; \
           $(RM) -f ex1_3.tmp
	-@$(MPIRUN) -np 2 ex1 -pcmethod sor -sor_symmetric -kspmonitor >\
           ex1_4.tmp 2>&1;   \
           if (diff ex1_4.out ex1_4.tmp) then true; \
           else echo "Possible problem with ex1_4, diffs above"; fi; \
           $(RM) -f ex1_4.tmp
	-@$(MPIRUN) -np 1 ex1 -pcmethod eisenstat -kspmonitor >\
           ex1_5.tmp 2>&1;   \
           if (diff ex1_5.out ex1_5.tmp) then true; \
           else echo "Possible problem with ex1_5, diffs above"; fi; \
           $(RM) -f ex1_5.tmp
	-@$(MPIRUN) -np 2 ex1 -pcmethod eisenstat -kspmonitor >\
           ex1_6.tmp 2>&1;   \
           if (diff ex1_6.out ex1_6.tmp) then true; \
           else echo "Possible problem with ex1_6, diffs above"; fi; \
           $(RM) -f ex1_6.tmp
	-@$(MPIRUN) ex2 -kspmonitor -m 5 > ex2_1.tmp 2>&1;   \
           if (diff ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi; \
           $(RM) -f ex2_1.tmp
	-@$(MPIRUN) -np 2 ex2 -kspmonitor -m 5 > ex2_2.tmp 2>&1;   \
           if (diff ex2_2.out ex2_2.tmp) then true; \
           else echo "Possible problem with ex2_2, diffs above"; fi; \
           $(RM) -f ex2_2.tmp
	-@$(MPIRUN) -np 1 ex2 -pcmethod sor -sor_symmetric -kspmonitor > \
            ex2_3.tmp 2>&1;   \
           if (diff ex2_3.out ex2_3.tmp) then true; \
           else echo "Possible problem with ex2_3, diffs above"; fi; \
           $(RM) -f ex2_3.tmp
	-@$(MPIRUN) -np 2 ex2 -pcmethod sor -sor_symmetric -kspmonitor >\
            ex2_4.tmp 2>&1;   \
           if (diff ex2_4.out ex2_4.tmp) then true; \
           else echo "Possible problem with ex2_4, diffs above"; fi; \
           $(RM) -f ex2_4.tmp
	-@$(MPIRUN) -np 1 ex2 -pcmethod eisenstat -kspmonitor >\
            ex2_5.tmp 2>&1;   \
           if (diff ex2_5.out ex2_5.tmp) then true; \
           else echo "Possible problem with ex2_5, diffs above"; fi; \
           $(RM) -f ex2_5.tmp
	-@$(MPIRUN) ex3 -kspmonitor -m 5 > ex3_1.tmp 2>&1;   \
           if (diff ex3_1.out ex3_1.tmp) then true; \
           else echo "Possible problem with ex3_1, diffs above"; fi; \
           $(RM) -f ex3_1.tmp
	-@$(MPIRUN) -np 2 ex3 -kspmonitor -m 5 > ex3_2.tmp 2>&1;   \
           if (diff ex3_2.out ex3_2.tmp) then true; \
           else echo "Possible problem with ex3_2, diffs above"; fi; \
           $(RM) -f ex3_2.tmp
	-@$(MPIRUN) -np 2 ex3 -pcmethod sor -sor_symmetric -kspmonitor >\
           ex3_3.tmp 2>&1;   \
           if (diff ex3_3.out ex3_3.tmp) then true; \
           else echo "Possible problem with ex3_3, diffs above"; fi; \
           $(RM) -f ex3_3.tmp
	-@$(MPIRUN) ex4 -kspmonitor -m 5 > ex4_1.tmp 2>&1;   \
           if (diff ex4_1.out ex4_1.tmp) then true; \
           else echo "Possible problem with ex4_1, diffs above"; fi; \
           $(RM) -f ex4_1.tmp
	-@$(MPIRUN) -np 2 ex4 -kspmonitor -m 5 > ex4_2.tmp 2>&1;   \
           if (diff ex4_2.out ex4_2.tmp) then true; \
           else echo "Possible problem with ex4_2, diffs above"; fi; \
           $(RM) -f ex4_2.tmp
	-@$(MPIRUN) -np 1 ex4 -pcmethod sor -sor_symmetric -kspmonitor -m 5 >\
           ex4_3.tmp 2>&1;   \
           if (diff ex4_3.out ex4_3.tmp) then true; \
           else echo "Possible problem with ex4_3, diffs above"; fi; \
           $(RM) -f ex4_3.tmp
	-@$(MPIRUN) -np 2 ex4 -pcmethod sor -sor_symmetric -kspmonitor -m 5 >\
           ex4_4.tmp 2>&1;   \
           if (diff ex4_4.out ex4_4.tmp) then true; \
           else echo "Possible problem with ex4_4, diffs above"; fi; \
           $(RM) -f ex4_4.tmp
	-@$(MPIRUN) -np 1 ex4 -pcmethod eisenstat -kspmonitor -m 5  >\
           ex4_5.tmp 2>&1;   \
           if (diff ex4_5.out ex4_5.tmp) then true; \
           else echo "Possible problem with ex4_5, diffs above"; fi; \
           $(RM) -f ex4_5.tmp
	-@$(MPIRUN) ex5 -kspmonitor -m 5 >\
           ex5_1.tmp 2>&1;   \
           if (diff ex5_1.out ex5_1.tmp) then true; \
           else echo "Possible problem with ex5_1, diffs above"; fi; \
           $(RM) -f ex5_1.tmp
	-@$(MPIRUN) -np 2 ex5 -kspmonitor -m 5 >\
           ex5_2.tmp 2>&1;   \
           if (diff ex5_2.out ex5_2.tmp) then true; \
           else echo "Possible problem with ex5_2, diffs above"; fi; \
           $(RM) -f ex5_2.tmp
	-@$(MPIRUN) ex6 -kspmonitor -m 5 >\
           ex6_1.tmp 2>&1;   \
           if (diff ex6_1.out ex6_1.tmp) then true; \
           else echo "Possible problem with ex6_1, diffs above"; fi; \
           $(RM) -f ex6_1.tmp
	-@$(MPIRUN) -np 2 ex6 -kspmonitor -m 5 >\
           ex6_2.tmp 2>&1;   \
           if (diff ex6_2.out ex6_2.tmp) then true; \
           else echo "Possible problem with ex6_2, diffs above"; fi; \
           $(RM) -f ex6_2.tmp
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

ex6: ex6.o 
	-$(CLINKER) -o ex6 ex6.o $(LIBNAME) $(SYSLIB)
	$(RM) ex6.o

ex7: ex7.o 
	-$(CLINKER) -o ex7 ex7.o $(LIBNAME) $(SYSLIB)
	$(RM) ex7.o

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
