#PETSC_DIR = ../../../..
ALL: ex1

CFLAGS	        = $(CPPFLAGS) -D__DIR__='"$(LOCDIR)"' $(BS_INCLUDE)
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        = pargrid.h
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscsles
LOCDIR          = src/sles/examples/tutorials/
TESTEXAMPLES_1  = ex1.PETSc runex1 runex1_2 runex1_3 \
		  ex1.rm ex2.PETSc runex2 runex2_2 runex2_3 runex2_4 \
		  ex2.rm ex7.PETSc runex7 ex7.rm ex5.PETSc runex5 \
		  runex5_2 ex5.rm ex9.PETSc runex9 ex9.rm clean
TESTEXAMPLES_2  = ex4.PETSc ex4.rm ex8.PETSc ex8.rm clean
TESTEXAMPLES_3  = ex1f.PETSc runex1f ex1f.rm ex2f.PETSc runex2f ex2f.rm ex6f.PETSc clean
TESTEXAMPLES_9  = ex1f.PETSc runex1f ex1f.rm ex6f.PETSc runex6f ex6f.rm clean
TESTEXAMPLES_4  = ex1.PETSc runex1 runex1_2 runex1_3 ex1.rm ex2.PETSc runex2 \
		  runex2_3 runex2_5 ex2.rm ex7.PETSc ex7.rm ex5.PETSc ex5.rm \
		  ex4.PETSc ex4.rm ex9.PETSc runex9 ex9.rm clean
TESTEXAMPLES_5  = ex11.PETSc runex11 runex11_2 runex11_3 ex11.rm clean
TESTEXAMPLES_6  = 
TESTEXAMPLES_7  = ex5.PETSc runex5_3 runex5_4 ex5.rm clean
TESTEXAMPLES_8  = 

BUILDEXAMPLES_1 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex7.PETSc ex7.rm \
		  ex5.PETSc ex5.rm ex9.PETSc ex9.rm clean
BUILDEXAMPLES_2 = ex4.PETSc ex4.rm ex8.PETSc ex8.rm clean
BUILDEXAMPLES_3 = ex1f.PETSc ex1f.rm ex2f.PETSc ex2f.rm ex6f.PETSc ex6f.rm clean
BUILDEXAMPLES_4 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex7.PETSc ex7.rm \
		  ex5.PETSc ex5.rm ex4.PETSc ex4.rm ex9.PETSc ex9.rm clean
BUILDEXAMPLES_5 = ex11.PETSc ex11.rm clean
BUILDEXAMPLES_6 = ex5.PETSc ex5.rm clean
BUILDEXAMPLES_7 = ex5.PETSc ex5.rm clean
BUILDEXAMPLES_8 = 
BUILDEXAMPLES_9 = ex1f.PETSc ex1f.rm ex20.PETSc ex20.rm clean

EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex7.c ex8.c ex9.c ex10.c ex11.c
EXAMPLESF       = ex1f.F ex2f.F ex6f.F

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) ex1.o

ex1f: ex1f.o  chkopts
	-$(FLINKER) -o ex1f ex1f.o  $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex1f.o

ex2: ex2.o  chkopts
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) ex2.o

ex2f: ex2f.o  chkopts
	-$(FLINKER) -o ex2f ex2f.o  $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex2f.o

ex3: ex3.o  chkopts
	-$(CLINKER) -o ex3 ex3.o  $(PETSC_LIB)
	$(RM) ex3.o

ex4: ex4.o  chkopts
	-$(CLINKER) -o ex4 ex4.o  $(PETSC_LIB)
	$(RM) ex4.o

ex5: ex5.o  chkopts
	-$(CLINKER) -o ex5 ex5.o  $(PETSC_LIB)
	$(RM) ex5.o

ex6f: ex6f.o  chkopts
	-$(FLINKER) -o ex6f ex6f.o  $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex6f.o

ex7: ex7.o  chkopts
	-$(CLINKER) -o ex7 ex7.o  $(PETSC_LIB)
	$(RM) ex7.o

ex8: ex8.o  chkopts
	-$(CLINKER) -o ex8 ex8.o  $(PETSC_LIB)
	$(RM) ex8.o

ex9: ex9.o  chkopts
	-$(CLINKER) -o ex9 ex9.o  $(PETSC_LIB)
	$(RM) ex9.o

ex10: ex10.o  chkopts
	-$(CLINKER) -o ex10 ex10.o  $(PETSC_LIB)
	$(RM) ex10.o

ex11: ex11.o  chkopts
	-$(CLINKER) -o ex11 ex11.o  $(PETSC_LIB)
	$(RM) ex11.o


runex1:
	-@$(MPIRUN) ex1 -ksp_smonitor > ex1_1.tmp 2>&1;	  \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp

runex1_2:
	-@$(MPIRUN) ex1 -pc_type sor -pc_sor_symmetric -ksp_smonitor >\
	   ex1_2.tmp 2>&1;   \
	   if (diff output/ex1_2.out ex1_2.tmp) then true; \
	   else echo "Possible problem with ex1_2, diffs above"; fi; \
	   $(RM) -f ex1_2.tmp

runex1_3:
	-@$(MPIRUN) ex1 -pc_type eisenstat -ksp_smonitor >\
	   ex1_3.tmp 2>&1;   \
	   if (diff output/ex1_3.out ex1_3.tmp) then true; \
	   else echo "Possible problem with ex1_3, diffs above"; fi; \
	   $(RM) -f ex1_3.tmp

runex1f:
	-@$(MPIRUN) ex1f -ksp_smonitor > ex1f_1.tmp 2>&1;	  \
	   if (diff output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   $(RM) -f ex1f_1.tmp

runex2:
	-@$(MPIRUN) ex2 -ksp_smonitor -m 5 -n 5 > ex2_1.tmp 2>&1; \
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   $(RM) -f ex2_1.tmp

runex2_2:
	-@$(MPIRUN) -np 2 ex2 -ksp_smonitor -m 5 -n 5 > ex2_2.tmp 2>&1; \
	   if (diff output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi; \
	   $(RM) -f ex2_2.tmp

runex2_3:
	-@$(MPIRUN) ex2 -pc_type sor -pc_sor_symmetric -ksp_smonitor > \
	    ex2_3.tmp 2>&1;   \
	   if (diff output/ex2_3.out ex2_3.tmp) then true; \
	   else echo "Possible problem with ex2_3, diffs above"; fi; \
	   $(RM) -f ex2_3.tmp

runex2_4:
	-@$(MPIRUN) ex2 -pc_type eisenstat -ksp_smonitor >\
	    ex2_4.tmp 2>&1;   \
	   if (diff output/ex2_4.out ex2_4.tmp) then true; \
	   else echo "Possible problem with ex2_4, diffs above"; fi; \
	   $(RM) -f ex2_4.tmp

runex2f:
	-@$(MPIRUN) -np 2 ex2f -pc_type jacobi -ksp_smonitor > ex2f_1.tmp 2>&1; \
	   if (diff output/ex2f_1.out ex2f_1.tmp) then true; \
	   else echo "Possible problem with ex2f_1, diffs above"; fi; \
	   $(RM) -f ex2f_1.tmp

runex4:
	-@$(MPIRUN) -np 3  ex4 -ksp_smonitor > ex4_1.tmp 2>&1; \
	   if (diff output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   $(RM) -f ex4_1.tmp

runex5:
	-@$(MPIRUN) ex5 -pc_type jacobi -ksp_smonitor > ex5_1.tmp 2>&1; \
	   if (diff output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   $(RM) -f ex5_1.tmp

runex5_2:
	-@$(MPIRUN) -np 2 ex5 -pc_type jacobi -ksp_smonitor \
	   -ksp_rtol .000001 > ex5_2.tmp 2>&1;   \
	   if (diff output/ex5_2.out ex5_2.tmp) then true; \
	   else echo "Possible problem with ex5_2, diffs above"; fi; \
	   $(RM) -f ex5_2.tmp

# Note: We test these BlockSolve95 examples differently from the usual
# because the coloring heuristic is machine-dependent; convergence rates
# may differ slightly on various architectures.
runex5_3:
	-@$(MPIRUN) -np 2 ex5 -pc_type icc -ksp_type cg -mat_mpirowbs -ksp_rtol .000001;\
	echo "ex5_3: This program tested the ICC preconditioner in BlockSolve95."

runex5_4:
	-@$(MPIRUN) -np 2 ex5 -pc_type ilu -mat_nonsym -mat_mpirowbs \
	   -ksp_rtol .000001;\
	echo "ex5_4: This program tested the ILU preconditioner in BlockSolve95."
runex6f:
	-@$(MPIRUN) ex6f -pc_type jacobi -mat_view -ksp_smonitor > ex6f_1.tmp 2>&1; \
	   if (diff output/ex6f_1.out ex6f_1.tmp) then true; \
	   else echo "Possible problem with ex6f_1, diffs above"; fi; \
	   $(RM) -f ex6f_1.tmp

runex7:
	-@$(MPIRUN) -np 2 ex7 -ksp_smonitor -noslesview> ex7_1.tmp 2>&1; \
	   if (diff output/ex7_1.out ex7_1.tmp) then true; \
	   else echo "Possible problem with ex7_1, diffs above"; fi; \
	   $(RM) -f ex7_1.tmp

runex9:
	-@$(MPIRUN) ex9 -t 2 -pc_type jacobi -ksp_smonitor -ksp_type gmres \
	   -s2_ksp_type bcgs -s2_pc_type jacobi -s2_ksp_smonitor \
           > ex9_1.tmp 2>&1; \
	   if (diff output/ex9_1.out ex9_1.tmp) then true; \
	   else echo "Possible problem with ex9_1, diffs above"; fi; \
	   $(RM) -f ex9_1.tmp
runex11:
	-@$(MPIRUN) ex11 -mat_aij -mat_view_info_detailed -pc_type jacobi -ksp_smonitor > ex11_1.tmp 2>&1; \
	   if (diff output/ex11_1.out ex11_1.tmp) then true; \
	   else echo "Possible problem with ex11_1, diffs above"; fi; \
	   $(RM) -f ex11_1.tmp

runex11_2:
	-@$(MPIRUN) -np 4 ex11 -m_eta 10 -m_xi 11 -mat_aij -mat_view_info \
	   -ksp_smonitor -pc_type bjacobi > ex11_2.tmp 2>&1;	  \
	   if (diff output/ex11_2.out ex11_2.tmp) then true; \
	   else echo "Possible problem with ex11_2, diffs above"; fi; \
	   $(RM) -f ex11_2.tmp

runex11_3:
	-@$(MPIRUN) ex11 -mat_aij -print_debug -modify_submat \
	   -ksp_smonitor -pc_type bjacobi -pc_bjacobi_blocks 4 > ex11_3.tmp 2>&1;	  \
	   if (diff output/ex11_3.out ex11_3.tmp) then true; \
	   else echo "Possible problem with ex11_3, diffs above"; fi; \
	   $(RM) -f ex11_3.tmp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
