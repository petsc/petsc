IPETSCDIR = ../../../
ALL: ex1

CFLAGS   = $(OPT) $(INCLUDEDIRS)$(CONF) $(COPT)
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
EXAMPLES = ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10 ex11 ex12 ex13 ex14\
           ex15 ex16 ex17 ex18 ex19 ex20 ex21 ex22
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c\
           ex10.c ex11.c ex12.c ex13.c ex14.c ex15.c ex16.c ex17.c ex18.c\
           ex19.c ex20.c ex21.c ex22.c
SYSLIB   = -L$(LDIR) -lpetscvec -lpetscsys -lpetscdraw\
           $(BS_LIB) $(LAPACK_LIB) $(BLAS_LIB) \
           $(MPI_LIB) $(SYS_LIB) $(FC_LIB) $(X11_LIB) -lm 
LINCLUDE = $(SOURCEH)

runexamples:
	-@echo "Examples in src/mat/examples -----------"
	-@$(MPIRUN)  ex1 > ex1_1.tmp 2>&1;   \
           if (diff ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           $(RM) -f ex1_1.tmp
	-@$(MPIRUN)  ex2 > ex2_1.tmp 2>&1;   \
           if (diff ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi; \
           $(RM) -f ex2_1.tmp
	-@$(MPIRUN)  ex3 > ex3_1.tmp 2>&1;   \
           if (diff ex3_1.out ex3_1.tmp) then true; \
           else echo "Possible problem with ex3_1, diffs above"; fi; \
           $(RM) -f ex3_1.tmp
	-@$(MPIRUN)  ex4 > ex4_1.tmp 2>&1;   \
           if (diff ex4_1.out ex4_1.tmp) then true; \
           else echo "Possible problem with ex4_1, diffs above"; fi; \
           $(RM) -f ex4_1.tmp
	-@$(MPIRUN)  ex5 > ex5_1.tmp 2>&1;   \
           if (diff ex5_1.out ex5_1.tmp) then true; \
           else echo "Possible problem with ex5_1, diffs above"; fi; \
           $(RM) -f ex5_1.tmp
	-@$(MPIRUN)  ex6 > ex6_1.tmp 2>&1;   \
           if (diff ex6_1.out ex6_1.tmp) then true; \
           else echo "Possible problem with ex6_1, diffs above"; fi; \
           $(RM) -f ex6_1.tmp
	-@$(MPIRUN)  ex7 > ex7_1.tmp 2>&1;   \
           if (diff ex7_1.out ex7_1.tmp) then true; \
           else echo "Possible problem with ex7_1, diffs above"; fi; \
           $(RM) -f ex7_1.tmp
	-@$(MPIRUN)  ex8 > ex8_1.tmp 2>&1;   \
           if (diff ex8_1.out ex8_1.tmp) then true; \
           else echo "Possible problem with ex8_1, diffs above"; fi; \
           $(RM) -f ex8_1.tmp
	-@$(MPIRUN)  -np 2 ex9 > ex9_1.tmp 2>&1;   \
           if (diff ex9_1.out ex9_1.tmp) then true; \
           else echo "Possible problem with ex9_1, diffs above"; fi; \
           $(RM) -f ex9_1.tmp
	-@$(MPIRUN)  ex10 > ex10_1.tmp 2>&1;   \
           if (diff ex10_1.out ex10_1.tmp) then true; \
           else echo "Possible problem with ex10_1, diffs above"; fi; \
           $(RM) -f ex10_1.tmp
	-@$(MPIRUN)  ex11 > ex11_1.tmp 2>&1;   \
           if (diff ex11_1.out ex11_1.tmp) then true; \
           else echo "Possible problem with ex11_1, diffs above"; fi; \
           $(RM) -f ex11_1.tmp
	-@$(MPIRUN)  -np 2 ex12 > ex12_1.tmp 2>&1;   \
           if (diff ex12_1.out ex12_1.tmp) then true; \
           else echo "Possible problem with ex12_1, diffs above"; fi; \
           $(RM) -f ex12_1.tmp
	-@$(MPIRUN)  ex13 > ex13_1.tmp 2>&1;   \
           if (diff ex13_1.out ex13_1.tmp) then true; \
           else echo "Possible problem with ex13_1, diffs above"; fi; \
           $(RM) -f ex13_1.tmp
	-@$(MPIRUN)  ex14 > ex14_1.tmp 2>&1;   \
           if (diff ex14_1.out ex14_1.tmp) then true; \
           else echo "Possible problem with ex14_1, diffs above"; fi; \
           $(RM) -f ex14_1.tmp
	-@$(MPIRUN)  ex15 > ex15_1.tmp 2>&1;   \
           if (diff ex15_1.out ex15_1.tmp) then true; \
           else echo "Possible problem with ex15_1, diffs above"; fi; \
           $(RM) -f ex15_1.tmp
	-@$(MPIRUN)  ex16 > ex16_1.tmp 2>&1;   \
           if (diff ex16_1.out ex16_1.tmp) then true; \
           else echo "Possible problem with ex16_1, diffs above"; fi; \
           $(RM) -f ex16_1.tmp
	-@$(MPIRUN)  -np 4 ex19 > ex19_1.tmp 2>&1;   \
           if (diff ex19_1.out ex19_1.tmp) then true; \
           else echo "Possible problem with ex19_1, diffs above"; fi; \
           $(RM) -f ex19_1.tmp
	-@$(MPIRUN)  ex20 > ex20_1.tmp 2>&1;   \
           if (diff ex20_1.out ex20_1.tmp) then true; \
           else echo "Possible problem with ex20_1, diffs above"; fi; \
           $(RM) -f ex21_1.tmp
	-@$(MPIRUN)  ex21 > ex21_1.tmp 2>&1;   \
           if (diff ex21_1.out ex21_1.tmp) then true; \
           else echo "Possible problem with ex21_1, diffs above"; fi; \
           $(RM) -f ex20_1.tmp
	-@$(MPIRUN)  ex22 > ex22_1.tmp 2>&1;   \
           if (diff ex22_1.out ex22_1.tmp) then true; \
           else echo "Possible problem with ex22_1, diffs above"; fi; \
           $(RM) -f ex22_1.tmp

ex1: ex1.o
	-$(CLINKER)  -o ex1 ex1.o $(LIBNAME) $(SYSLIB)
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

ex8: ex8.o
	-$(CLINKER) -o ex8 ex8.o $(LIBNAME) $(SYSLIB)
	$(RM) ex8.o

ex9: ex9.o
	-$(CLINKER) -o ex9 ex9.o $(LIBNAME) $(SYSLIB)
	$(RM) ex9.o

ex10: ex10.o
	-$(CLINKER) -o ex10 ex10.o $(LIBNAME) $(SYSLIB)
	$(RM) ex10.o

ex11: ex11.o
	-$(CLINKER) -o ex11 ex11.o $(LIBNAME) $(SYSLIB)
	$(RM) ex11.o

ex12: ex12.o
	-$(CLINKER) -o ex12 ex12.o $(LIBNAME) $(SYSLIB)
	$(RM) ex12.o

ex13: ex13.o
	-$(CLINKER) -o ex13 ex13.o $(LIBNAME) $(SYSLIB)
	$(RM) ex13.o

ex14: ex14.o
	-$(CLINKER) -o ex14 ex14.o $(LIBNAME) $(SYSLIB)
	$(RM) ex14.o

ex15: ex15.o
	-$(CLINKER) -o ex15 ex15.o $(LIBNAME) $(SYSLIB)
	$(RM) ex15.o

ex16: ex16.o
	-$(CLINKER) -o ex16 ex16.o $(LIBNAME) $(SYSLIB)
	$(RM) ex16.o

ex17: ex17.o
	-$(CLINKER) -o ex17 ex17.o $(LIBNAME) $(SYSLIB)
	$(RM) ex17.o

ex18: ex18.o
	-$(CLINKER) -o ex18 ex18.o $(LIBNAME) $(SYSLIB)
	$(RM) ex18.o

ex19: ex19.o
	-$(CLINKER) -o ex19 ex19.o $(LIBNAME) $(SYSLIB)
	$(RM) ex19.o

ex20: ex20.o
	-$(CLINKER) -o ex20 ex20.o $(LIBNAME) $(SYSLIB)
	$(RM) ex20.o

ex21: ex21.o
	-$(CLINKER) -o ex21 ex21.o $(LIBNAME) $(SYSLIB)
	$(RM) ex21.o

ex22: ex22.o
	-$(CLINKER) -o ex22 ex22.o $(LIBNAME) $(SYSLIB)
	$(RM) ex22.o

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)
