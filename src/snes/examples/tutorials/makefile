#PETSC_DIR = ../../../..
ALL: ex1

CFLAGS	        = $(CPPFLAGS) -D__DIR__='"$(LOCDIR)"'
FFLAGS	        =	 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscsnes
LOCDIR          = src/snes/examples/tutorials/
TESTEXAMPLES_1  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm \
		  ex3.PETSc runex3 runex3_2 runex3_3 ex3.rm ex6.PETSc runex6 runex6_2 \
		  ex6.rm ex4.PETSc runex4 runex4_2 ex4.rm ex5.PETSc runex5\
		  runex5_2 runex5_3 runex5_4 runex5_5 runex5_6 runex5_7 \
                  ex5.rm clean
TESTEXAMPLES_2  = ex3.PETSc runex3 runex3_2 runex3_3 ex3.rm clean
TESTEXAMPLES_3  = ex1f.PETSc runex1f ex1f.rm clean
TESTEXAMPLES_8  = ex5f.PETSc runex5f runex5f_2 runex5f_3 ex5f.rm clean
TESTEXAMPLES_9  = ex1f.PETSc runex1f ex1f.rm clean
TESTEXAMPLES_4  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc\
		  runex3 ex3.rm ex6.PETSc runex6 runex6_2 ex6.rm \
		  ex4.PETSc runex4 runex4_2 ex4.rm ex5.PETSc ex5.rm
BUILDEXAMPLES_1 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm ex6.PETSc \
		  ex6.rm ex4.PETSc ex4.rm ex5.PETSc ex5.rm clean
BUILDEXAMPLES_2 = ex3.PETSc ex3.rm clean
BUILDEXAMPLES_3 = ex1f.PETSc ex1f.rm clean
BUILDEXAMPLES_4 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm ex6.PETSc \
		  ex6.rm ex4.PETSc ex4.rm ex5.PETSc ex5.rm clean
BUILDEXAMPLES_8 = ex5f.PETSc ex5f.rm clean

EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c
EXAMPLESF       = ex1f.F ex5f.F
DIRS	        = 

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o $(PETSC_LIB)
	$(RM) ex1.o

ex1f: ex1f.o  chkopts
	-$(FLINKER) -o ex1f ex1f.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex1f.o

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

ex5f: ex5f.o  chkopts
	-$(FLINKER) -o ex5f ex5f.o $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex5f.o

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

runex1:
	-@$(MPIRUN) ex1 -snes_smonitor > ex1_1.tmp 2>&1;   \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp
runex1f:
	-@$(MPIRUN) ex1f -snes_smonitor > ex1f_1.tmp 2>&1;   \
	   if (diff output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   $(RM) -f ex1f_1.tmp
runex2:
	-@$(MPIRUN) ex2 -nox -snes_smonitor -snes_view \
		-pc_type jacobi >ex2_1.tmp 2>&1;	\
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   $(RM) -f ex2_1.tmp
runex2_2:
	-@$(MPIRUN) ex2 -nox -snes_smonitor -snes_type tr -snes_view \
		-pc_type jacobi > ex2_2.tmp 2>&1; \
	   if (diff output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi; \
	   $(RM) -f ex2_2.tmp
runex3:
	-@$(MPIRUN) ex3 -nox -snes_smonitor  > ex3_1.tmp 2>&1;	  \
	   if (diff output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   $(RM) -f ex3_1.tmp
runex3_2:
	-@$(MPIRUN) -np 3 ex3 -nox -pc_type asm  -mat_mpiaij -mat_aij_oneindex \
	   -snes_smonitor > ex3_2.tmp 2>&1; \
	   if (diff output/ex3_2.out ex3_2.tmp) then true; \
	   else echo "Possible problem with ex3_2, diffs above"; fi; \
	   $(RM) -f ex3_2.tmp
runex3_3:
	-@$(MPIRUN) -np 2 ex3 -nox -snes_smonitor > ex3_3.tmp 2>&1; \
	   if (diff output/ex3_3.out ex3_3.tmp) then true; \
	   else echo "Possible problem with ex3_3, diffs above"; fi; \
	   $(RM) -f ex3_3.tmp
runex4:
	-@$(MPIRUN) ex4 -nox > ex4_1.tmp 2>&1;	 \
	   if (diff output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   $(RM) -f ex4_1.tmp
runex4_2:
	-@$(MPIRUN) ex4 -snes_smonitor -nox -snes_type tr > ex4_2.tmp 2>&1;\
	   if (diff output/ex4_2.out ex4_2.tmp) then true; \
	   else echo "Possible problem with ex4_2, diffs above"; fi; \
	   $(RM) -f ex4_2.tmp
runex5:
	-@$(MPIRUN) -np 4 ex5 -snes_mf -Nx 4 -Ny 1 -snes_smonitor > ex5_1.tmp 2>&1; \
	   if (diff output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   $(RM) -f ex5_1.tmp
runex5_2:
	-@$(MPIRUN) -np 4 ex5 -mat_mpiaij -Nx 2 -Ny 2 -snes_smonitor > ex5_2.tmp 2>&1; \
	   if (diff output/ex5_2.out ex5_2.tmp) then true; \
	   else echo "Possible problem with ex5_2, diffs above"; fi; \
	   $(RM) -f ex5_2.tmp
runex5_3:
	-@$(MPIRUN) -np 4 ex5 -snes_mf -Nx 2 -Ny 2 -star -snes_smonitor>ex5_3.tmp 2>&1;\
	   if (diff output/ex5_1.out ex5_3.tmp) then true; \
	   else echo "Possible problem with ex5_3, diffs above"; fi; \
	   $(RM) -f ex5_3.tmp
runex5_4:
	-@$(MPIRUN) -np 4 ex5 -mat_mpiaij -Nx 2 -Ny 2 -snes_smonitor -pc_type asm \
	   -pc_asm_blocks 4 -pc_asm_overlap 1 > ex5_4.tmp 2>&1; \
	   if (diff output/ex5_4.out ex5_4.tmp) then true; \
	   else echo "Possible problem with ex5_4, diffs above"; fi; \
	   $(RM) -f ex5_4.tmp
runex5_5:
	-@$(MPIRUN) -np 4 ex5 -mat_mpiaij -Nx 2 -Ny 2 -snes_smonitor -pc_type bjacobi \
	   -snes_view > ex5_5.tmp 2>&1; \
	   if (diff output/ex5_5.out ex5_5.tmp) then true; \
	   else echo "Possible problem with ex5_5, diffs above"; fi; \
	   $(RM) -f ex5_5.tmp
runex5_6:
	-@$(MPIRUN) -np 4 ex5 -snes_fd -Nx 2 -Ny 2 -snes_smonitor > ex5_6.tmp 2>&1; \
	   if (diff output/ex5_6.out ex5_6.tmp) then true; \
	   else echo "Possible problem with ex5_6, diffs above"; fi; \
	   $(RM) -f ex5_6.tmp
runex5_7:
	-@$(MPIRUN) -np 3 ex5 -snes_fd  -snes_smonitor > ex5_7.tmp 2>&1;\
	   if (diff output/ex5_7.out ex5_7.tmp) then true; \
	   else echo "Possible problem with ex5_7, diffs above"; fi; \
	   $(RM) -f ex5_7.tmp
runex5f:
	-@$(MPIRUN) -np 4 ex5f -snes_mf -Nx 4 -Ny 1 -snes_smonitor > ex5f_1.tmp 2>&1; \
	   if (diff output/ex5f_1.out ex5f_1.tmp) then true; \
	   else echo "Possible problem with ex5f_1, diffs above"; fi; \
	   $(RM) -f ex5f_1.tmp
runex5f_2:
	-@$(MPIRUN) -np 4 ex5f -mat_mpiaij -Nx 2 -Ny 2 -snes_smonitor > ex5f_2.tmp 2>&1; \
	   if (diff output/ex5f_2.out ex5f_2.tmp) then true; \
	   else echo "Possible problem with ex5f_2, diffs above"; fi; \
	   $(RM) -f ex5f_2.tmp
runex5f_3:
	-@$(MPIRUN) -np 3 ex5f -snes_fd  -snes_smonitor > ex5f_3.tmp 2>&1;\
	   if (diff output/ex5f_3.out ex5f_3.tmp) then true; \
	   else echo "Possible problem with ex5f_3, diffs above"; fi; \
	   $(RM) -f ex5f_3.tmp
runex6:
	-@$(MPIRUN) ex6 -snes_smonitor > ex6_1.tmp 2>&1;   \
	   if (diff output/ex6_1.out ex6_1.tmp) then true; \
	   else echo "Possible problem with ex6_1, diffs above"; fi; \
	   $(RM) -f ex6_1.tmp
runex6_2:
	-@$(MPIRUN) ex6 -snes_mf -snes_smonitor > ex6_2.tmp 2>&1;   \
	   if (diff output/ex6_2.out ex6_2.tmp) then true; \
	   else echo "Possible problem with ex6_2, diffs above"; fi; \
	   $(RM) -f ex6_2.tmp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
