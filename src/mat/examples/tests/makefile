PETSC_DIR = ../../..
ALL: ex1

CFLAGS	      =	 $(PETSC_INCLUDE) $(PCONF) $(COPT)
FFLAGS	      = 
SOURCEC	      = 
SOURCEF	      =
SOURCEH	      =
OBJSC	      =
OBJSF	      =
LIBBASE	      = libpetscmat
EXAMPLES_1    = ex1 ex3 ex5 ex6 ex8 ex9 ex10 ex11 ex14\
		ex15 ex16 ex19 ex20 ex21 ex22
RUNEXAMPLES_1 = runex1	runex3	runex5 runex6  runex8 runex9\
		runex10 runex11 runex14 runex15 runex15_2 runex16\
		runex20 runex21 runex22
EXAMPLES_2    = ex2 ex4 ex7 ex12 ex13 ex17 ex19 ex24 ex25 \
		ex27 ex28 ex29 ex30 ex31 ex32 ex33 ex34 ex35 ex37 ex38 ex40 ex41
RUNEXAMPLES_2 = runex2 runex2_2 runex4 runex7 runex12 runex13 runex19 \
		runex35 runex40
EXAMPLESC     = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c\
		ex10.c ex11.c ex12.c ex13.c ex14.c ex15.c ex16.c ex17.c \
		ex19.c ex20.c ex21.c ex22.c ex24.c ex25.c \
		ex27.c ex28.c ex29.c ex30.c ex31.c ex32.c ex33.c \
		ex34.c ex35.c ex37.c ex38.c ex39.c
EXAMPLESF     = ex36.F
EXAMPLES_3    = ex36
RUNEXAMPLES_3 = runex36
EXAMPLES_4    = ex1 ex2 ex4 ex5 ex6 ex10 ex14\
		ex15 ex16 ex20 ex21 ex35
RUNEXAMPLES_4 = runex1 runex2 runex2_2	runex4 runex5 runex6 \
		runex10 runex14 runex15 runex15_2 \
		runex16 runex20 runex21 runex35

ex1: ex1.o  chkopts
	-$(CLINKER)  -o ex1 ex1.o $(PETSC_LIB)
	$(RM) ex1.o

ex2: ex2.o chkopts
	-$(CLINKER) -o ex2 ex2.o $(PETSC_LIB)
	$(RM) ex2.o

ex3: ex3.o chkopts
	-$(CLINKER) -o ex3 ex3.o $(PETSC_LIB)
	$(RM) ex3.o

ex4: ex4.o chkopts
	-$(CLINKER) -o ex4 ex4.o $(PETSC_LIB)
	$(RM) ex4.o

ex5: ex5.o chkopts
	-$(CLINKER)  -o ex5 ex5.o $(PETSC_LIB)
	$(RM) ex5.o

ex6: ex6.o chkopts
	-$(CLINKER) -o ex6 ex6.o $(PETSC_LIB)
	$(RM) ex6.o

ex7: ex7.o chkopts
	-$(CLINKER) -o ex7 ex7.o $(PETSC_LIB)
	$(RM) ex7.o

ex8: ex8.o chkopts
	-$(CLINKER) -o ex8 ex8.o $(PETSC_LIB)
	$(RM) ex8.o

ex9: ex9.o chkopts
	-$(CLINKER) -o ex9 ex9.o $(PETSC_LIB)
	$(RM) ex9.o

ex10: ex10.o chkopts
	-$(CLINKER) -o ex10 ex10.o $(PETSC_LIB)
	$(RM) ex10.o

ex11: ex11.o chkopts
	-$(CLINKER) -o ex11 ex11.o $(PETSC_LIB)
	$(RM) ex11.o

ex12: ex12.o chkopts
	-$(CLINKER) -o ex12 ex12.o $(PETSC_LIB)
	$(RM) ex12.o

ex13: ex13.o chkopts
	-$(CLINKER) -o ex13 ex13.o $(PETSC_LIB)
	$(RM) ex13.o

ex14: ex14.o chkopts
	-$(CLINKER) -o ex14 ex14.o $(PETSC_LIB)
	$(RM) ex14.o

ex15: ex15.o chkopts
	-$(CLINKER) -o ex15 ex15.o $(PETSC_LIB)
	$(RM) ex15.o

ex16: ex16.o chkopts
	-$(CLINKER) -o ex16 ex16.o $(PETSC_LIB)
	$(RM) ex16.o

ex17: ex17.o chkopts
	-$(CLINKER) -o ex17 ex17.o $(PETSC_LIB)
	$(RM) ex17.o

ex18: ex18.o chkopts
	-$(CLINKER) -o ex18 ex18.o $(PETSC_LIB)
	$(RM) ex18.o

ex19: ex19.o chkopts
	-$(CLINKER) -o ex19 ex19.o $(PETSC_LIB)
	$(RM) ex19.o

ex20: ex20.o chkopts
	-$(CLINKER) -o ex20 ex20.o $(PETSC_LIB)
	$(RM) ex20.o

ex21: ex21.o chkopts
	-$(CLINKER) -o ex21 ex21.o $(PETSC_LIB)
	$(RM) ex21.o

ex22: ex22.o chkopts
	-$(CLINKER) -o ex22 ex22.o $(PETSC_LIB)
	$(RM) ex22.o

ex23: ex23.o chkopts
	-$(CLINKER) -o ex23 ex23.o $(PETSC_LIB)
	$(RM) ex23.o

ex24: ex24.o chkopts
	-$(CLINKER) -o ex24 ex24.o $(PETSC_LIB)
	$(RM) ex24.o

ex25: ex25.o chkopts
	-$(CLINKER) -o ex25 ex25.o $(PETSC_LIB)
	$(RM) ex25.o

ex26: ex26.o chkopts
	-$(CLINKER) -o ex26 ex26.o $(PETSC_LIB)
	$(RM) ex26.o

ex27: ex27.o chkopts
	-$(CLINKER) -o ex27 ex27.o $(PETSC_LIB)
	$(RM) ex27.o

ex28: ex28.o chkopts
	-$(CLINKER) -o ex28 ex28.o $(PETSC_LIB)
	$(RM) ex28.o

ex29: ex29.o chkopts
	-$(CLINKER) -o ex29 ex29.o $(PETSC_LIB)
	$(RM) ex29.o

ex30: ex30.o chkopts
	-$(CLINKER) -o ex30 ex30.o $(PETSC_LIB)
	$(RM) ex30.o

ex31: ex31.o chkopts
	-$(CLINKER) -o ex31 ex31.o $(PETSC_LIB)
	$(RM) ex31.o

ex32: ex32.o chkopts
	-$(CLINKER) -o ex32 ex32.o $(PETSC_LIB)
	$(RM) ex32.o

ex33: ex33.o chkopts
	-$(CLINKER) -o ex33 ex33.o $(PETSC_LIB)
	$(RM) ex33.o

ex34: ex34.o chkopts
	-$(CLINKER) -o ex34 ex34.o $(PETSC_LIB)
	$(RM) ex34.o

ex35: ex35.o chkopts
	-$(CLINKER) -o ex35 ex35.o $(PETSC_LIB)
	$(RM) ex35.o

ex36: ex36.o chkopts
	-$(FLINKER) -o ex36 ex36.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex36.o

ex37: ex37.o chkopts
	-$(CLINKER) -o ex37 ex37.o $(PETSC_LIB)
	$(RM) ex37.o

ex38: ex38.o chkopts
	-$(CLINKER) -o ex38 ex38.o $(PETSC_LIB)
	$(RM) ex38.o

ex39: ex39.o chkopts
	-$(CLINKER) -o ex39 ex39.o $(PETSC_LIB)
	$(RM) ex39.o

ex40: ex40.o chkopts
	-$(CLINKER) -o ex40 ex40.o $(PETSC_LIB)
	$(RM) ex40.o

ex41: ex41.o chkopts
	-$(CLINKER) -o ex41 ex41.o $(PETSC_LIB)
	$(RM) ex41.o

runex1:
	-@$(MPIRUN)  ex1 > ex1_1.tmp 2>&1;   \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp
runex2:
	-@$(MPIRUN)  ex2 > ex2_1.tmp 2>&1;   \
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   $(RM) -f ex2_1.tmp
runex2_2:
	-@$(MPIRUN)  ex2 -mat_row > ex2_2.tmp 2>&1;   \
	   if (diff output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi; \
	   $(RM) -f ex2_2.tmp
runex2_3:
	-@$(MPIRUN)  ex2 -mat_bdiag> ex2_3.tmp 2>&1;   \
	   if (diff output/ex2_3.out ex2_3.tmp) then true; \
	   else echo "Possible problem with ex2_3, diffs above"; fi; \
	   $(RM) -f ex2_3.tmp
runex3:
	-@$(MPIRUN)  ex3 > ex3_1.tmp 2>&1;   \
	   if (diff output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   $(RM) -f ex3_1.tmp
runex4:
	-@$(MPIRUN)  ex4 > ex4_1.tmp 2>&1;   \
	   if (diff output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   $(RM) -f ex4_1.tmp
runex5:
	-@$(MPIRUN)  ex5 > ex5_1.tmp 2>&1;   \
	   if (diff output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   $(RM) -f ex5_1.tmp
runex6:
	-@$(MPIRUN)  ex6 > ex6_1.tmp 2>&1;   \
	   if (diff output/ex6_1.out ex6_1.tmp) then true; \
	   else echo "Possible problem with ex6_1, diffs above"; fi; \
	   $(RM) -f ex6_1.tmp
runex7:
	-@$(MPIRUN)  ex7 > ex7_1.tmp 2>&1;   \
	   if (diff output/ex7_1.out ex7_1.tmp) then true; \
	   else echo "Possible problem with ex7_1, diffs above"; fi; \
	   $(RM) -f ex7_1.tmp
runex8:
	-@$(MPIRUN)  ex8 > ex8_1.tmp 2>&1;   \
	   if (diff output/ex8_1.out ex8_1.tmp) then true; \
	   else echo "Possible problem with ex8_1, diffs above"; fi; \
	   $(RM) -f ex8_1.tmp
runex9:
	-@$(MPIRUN)  -np 2 ex9 > ex9_1.tmp 2>&1;   \
	   if (diff output/ex9_1.out ex9_1.tmp) then true; \
	   else echo "Possible problem with ex9_1, diffs above"; fi; \
	   $(RM) -f ex9_1.tmp
runex10:
	-@$(MPIRUN)  ex10 > ex10_1.tmp 2>&1;   \
	   if (diff output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Possible problem with ex10_1, diffs above"; fi; \
	   $(RM) -f ex10_1.tmp
runex11:
	-@$(MPIRUN)  ex11 > ex11_1.tmp 2>&1;   \
	   if (diff output/ex11_1.out ex11_1.tmp) then true; \
	   else echo "Possible problem with ex11_1, diffs above"; fi; \
	   $(RM) -f ex11_1.tmp
runex12:
	-@$(MPIRUN)  -np 2 ex12 > ex12_1.tmp 2>&1;   \
	   if (diff output/ex12_1.out ex12_1.tmp) then true; \
	   else echo "Possible problem with ex12_1, diffs above"; fi; \
	   $(RM) -f ex12_1.tmp
runex13:
	-@$(MPIRUN)  ex13 > ex13_1.tmp 2>&1;   \
	   if (diff output/ex13_1.out ex13_1.tmp) then true; \
	   else echo "Possible problem with ex13_1, diffs above"; fi; \
	   $(RM) -f ex13_1.tmp
runex14:
	-@$(MPIRUN)  ex14 > ex14_1.tmp 2>&1;   \
	   if (diff output/ex14_1.out ex14_1.tmp) then true; \
	   else echo "Possible problem with ex14_1, diffs above"; fi; \
	   $(RM) -f ex14_1.tmp
runex15:
	-@$(MPIRUN)  ex15  > ex15_1.tmp 2>&1;	\
	   if (diff output/ex15_1.out ex15_1.tmp) then true; \
	   else echo "Possible problem with ex15_1, diffs above"; fi; \
	   $(RM) -f ex15_1.tmp
runex15_2: # Note that runex15 and runex15_2 have identical output
	-@$(MPIRUN)  ex15 -mat_mpirow  > ex15_2.tmp 2>&1;   \
	   if (diff output/ex15_1.out ex15_2.tmp) then true; \
	   else echo "Possible problem with ex15_2, diffs above"; fi; \
	   $(RM) -f ex15_2.tmp
runex16:
	-@$(MPIRUN)  ex16 > ex16_1.tmp 2>&1;   \
	   if (diff output/ex16_1.out ex16_1.tmp) then true; \
	   else echo "Possible problem with ex16_1, diffs above"; fi; \
	   $(RM) -f ex16_1.tmp
runex17:
	-@$(MPIRUN)  ex17 > ex17_1.tmp 2>&1;   \
	   if (diff output/ex17_1.out ex17_1.tmp) then true; \
	   else echo "Possible problem with ex17_1, diffs above"; fi; \
	   $(RM) -f ex17_1.tmp
runex19:
	-@$(MPIRUN)  -np 4 ex19 > ex19_1.tmp 2>&1;   \
	   if (diff output/ex19_1.out ex19_1.tmp) then true; \
	   else echo "Possible problem with ex19_1, diffs above"; fi; \
	   $(RM) -f ex19_1.tmp
runex20:
	-@$(MPIRUN)  ex20 -mat_row -conv_mat_bdiag > ex20_1.tmp 2>&1;	\
	   if (diff output/ex20_1.out ex20_1.tmp) then true; \
	   else echo "Possible problem with ex20_1, diffs above"; fi; \
	   $(RM) -f ex20_1.tmp
runex21:
	-@$(MPIRUN)  ex21 > ex21_1.tmp 2>&1;   \
	   if (diff output/ex21_1.out ex21_1.tmp) then true; \
	   else echo "Possible problem with ex21_1, diffs above"; fi; \
	   $(RM) -f ex21_1.tmp
runex22:
	-@$(MPIRUN)  ex22 > ex22_1.tmp 2>&1;   \
	   if (diff output/ex22_1.out ex22_1.tmp) then true; \
	   else echo "Possible problem with ex22_1, diffs above"; fi; \
	   $(RM) -f ex22_1.tmp
runex30:
	-@$(MPIRUN)  ex30 -nox > ex30_1.tmp 2>&1;   \
	   if (diff output/ex30_1.out ex30_1.tmp) then true; \
	   else echo "Possible problem with ex30_1, diffs above"; fi; \
	   $(RM) -f ex30_1.tmp
runex35:
	-@$(MPIRUN)  ex35 > ex35_1.tmp	2>&1;	\
	   if (diff output/ex35_1.out ex35_1.tmp) then true; \
	   else echo "Possible problem with ex35_1, diffs above"; fi; \
	   $(RM) -f ex35_1.tmp
runex36:
	-@$(MPIRUN)  ex36 > ex36_1.tmp 2>&1;   \
	   if (diff output/ex36_1.out ex36_1.tmp) then true; \
	   else echo "Possible problem with ex36_1, diffs above"; fi; \
	   $(RM) -f ex36_1.tmp
runex39:
	-@$(MPIRUN)  ex39 -m 7 > ex39_1.tmp   2>&1;   \
	   if (diff output/ex39_1.out ex39_1.tmp) then true; \
	   else echo "Possible problem with ex39_1, diffs above"; fi; \
	   $(RM) -f ex39_1.tmp
runex40:
	-@$(MPIRUN)  -np 3 ex40 -f matrices/arco1 -nd 7	 -ov 2 > ex40_1.tmp 2>&1; \
	   if (diff output/ex40_1.out ex40_1.tmp) then true; \
	   else echo "Possible problem with ex40_1, diffs above"; fi; \
	   $(RM) -f ex40_1.tmp
runex41:
	-@$(MPIRUN) -np 3 ex41 -f matrices/arco1 -nd 3	-ov 1 > ex41_1.tmp  2>&1; \
	   if (diff output/ex41_1.out ex41_1.tmp) then true; \
	   else echo "Possible problem with ex41_1, diffs above"; fi; \
	   $(RM) -f ex41_1.tmp
include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)



