PETSC_DIR = ../../..
ALL: ex1

CFLAGS	      =	 $(PCONF) $(PETSC_INCLUDE)
FFLAGS	      =	 
SOURCEC	      = 
SOURCEF	      =
SOURCEH	      =
OBJSC	      =
OBJSF	      =
LIBBASE	      = libpetscsnes
EXAMPLES_1    = ex3 ex4 ex5 ex6
RUNEXAMPLES_1 = runex3 runex3_2 runex4 runex4_2 runex5 runex5_2 runex6 \
		runex6_2 runex6_3
EXAMPLES_2    = ex1 ex2 ex7 ex8 ex9
RUNEXAMPLES_2 = runex1 runex2 runex7 runex8 runex9
EXAMPLES_3    = ex12
RUNEXAMPLES_3 = runex12
EXAMPLES_4    = ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9
RUNEXAMPLES_4 = runex1 runex2 runex3 runex3_2 runex4 runex4_2 runex5 runex5_2 \
		runex6_2 runex7 
EXAMPLESC     = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c
EXAMPLESF     = ex12.F
DIRS	      = umin
#
#  runex9_2 and runex9_3 are not included in RUNEXAMPLES_2 because 
# they require too many processors
#
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

ex8: ex8.o  chkopts
	-$(CLINKER) -o ex8 ex8.o $(PETSC_LIB)
	$(RM) ex8.o

ex9: ex9.o  chkopts
	-$(CLINKER) -o ex9 ex9.o $(PETSC_LIB)
	$(RM) ex9.o

ex10: ex10.o  chkopts
	-$(CLINKER) -o ex10 ex10.o $(PETSC_LIB)
	$(RM) ex10.o

ex11: ex11.o  chkopts
	-$(CLINKER) -o ex11 ex11.o $(PETSC_LIB)
	$(RM) ex11.o

ex12: ex12.o  chkopts
	-$(FLINKER) -o ex12 ex12.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex12.o

runex1:
	-@$(MPIRUN) ex1 > ex1_1.tmp 2>&1;   \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp

runex2:
	-@$(MPIRUN) ex2	 > ex2_1.tmp 2>&1;   \
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   $(RM) -f ex2_1.tmp

runex3:
	-@$(MPIRUN) ex3 -nox -snes_smonitor > ex3_1.tmp 2>&1;	\
	   if (diff output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   $(RM) -f ex3_1.tmp
runex3_2:
	-@$(MPIRUN) ex3 -nox -snes_smonitor -snes_method tr > ex3_2.tmp 2>&1; \
	   if (diff output/ex3_2.out ex3_2.tmp) then true; \
	   else echo "Possible problem with ex3_2, diffs above"; fi; \
	   $(RM) -f ex3_2.tmp
runex4:
	-@$(MPIRUN) ex4 -nox > ex4_1.tmp 2>&1;	 \
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
	-@$(MPIRUN) ex5 -snes_mf -snes_smonitor > ex5_2.tmp 2>&1;   \
	   if (diff output/ex5_2.out ex5_2.tmp) then true; \
	   else echo "Possible problem with ex5_2, diffs above"; fi; \
	   $(RM) -f ex5_2.tmp
runex6:
	-@$(MPIRUN) -np 4 ex6 -matrix_freeJ -Nx 4 -Ny 1 -snes_smonitor > ex6_1.tmp 2>&1;   \
	   if (diff output/ex6_1.out ex6_1.tmp) then true; \
	   else echo "Possible problem with ex6_1, diffs above"; fi; \
	   $(RM) -f ex6_1.tmp
runex6_2:
	-@$(MPIRUN) -np 4 ex6 -mat_mpiaij -Nx 2 -Ny 2 -snes_smonitor > ex6_2.tmp 2>&1;	 \
	   if (diff output/ex6_2.out ex6_2.tmp) then true; \
	   else echo "Possible problem with ex6_3, diffs above"; fi; \
	   $(RM) -f ex6_2.tmp
runex6_3:
	-@$(MPIRUN) -np 4 ex6 -matrix_freeJ -Nx 2 -Ny 2 -star -snes_smonitor>ex6_3.tmp 2>&1;\
	   if (diff output/ex6_1.out ex6_3.tmp) then true; \
	   else echo "Possible problem with ex6_3, diffs above"; fi; \
	   $(RM) -f ex6_3.tmp

runex7:
	-@$(MPIRUN) ex7 -snes_smonitor	> ex7_1.tmp 2>&1;   \
	   if (diff output/ex7_1.out ex7_1.tmp) then true; \
	   else echo "Possible problem with ex7_1, diffs above"; fi; \
	   $(RM) -f ex7_1.tmp
runex8:
	-@$(MPIRUN) -np 2 ex8 -snes_smonitor  > ex8_1.tmp 2>&1;	  \
	   if (diff output/ex8_1.out ex8_1.tmp) then true; \
	   else echo "Possible problem with ex8_1, diffs above"; fi; \
	   $(RM) -f ex8_1.tmp

runex9:
	-@$(MPIRUN) -np 2 ex9 -snes_smonitor  > ex9_1.tmp 2>&1;	  \
	   if (diff output/ex9_1.out ex9_1.tmp) then true; \
	   else echo "Possible problem with ex9_1, diffs above"; fi; \
	   $(RM) -f ex9_1.tmp

runex9_2:
	-@$(MPIRUN) -np 8 ex9 -snes_smonitor  > ex9_2.tmp 2>&1;	  \
	   if (diff output/ex9_1.out ex9_2.tmp) then true; \
	   else echo "Possible problem with ex9_1, diffs above"; fi; \
	   $(RM) -f ex9_2.tmp

runex9_3:
	-@$(MPIRUN) -np 8 ex9 -star -snes_smonitor  > ex9_3.tmp 2>&1;	\
	   if (diff output/ex9_1.out ex9_3.tmp) then true; \
	   else echo "Possible problem with ex9_1, diffs above"; fi; \
	   $(RM) -f ex9_3.tmp

runex12:
	-@$(MPIRUN) -np 2 ex12 -n 10 -snes_smonitor  > ex12_1.tmp 2>&1;	  \
	   if (diff output/ex12_1.out ex12_1.tmp) then true; \
	   else echo "Possible problem with ex12_1, diffs above"; fi; \
	   $(RM) -f ex12_1.tmp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
