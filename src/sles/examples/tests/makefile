PETSC_DIR = ../../..
ALL: ex1

CFLAGS	      = $(PCONF) $(PETSC_INCLUDE) $(BS_INCLUDE)
FFLAGS	      = 
SOURCEC	      = 
SOURCEF	      =
SOURCEH	      = pargrid.h
OBJSC	      =
OBJSF	      =
LIBBASE	      = libpetscsles
EXAMPLES_1    = ex1 ex2 ex3 ex4 ex5 ex8 ex19
RUNEXAMPLES_1 = runex1 runex1_2 runex1_3 runex1_4 runex1_5 runex1_6\
		runex2 runex2_2 runex2_3 runex2_4 runex2_5\
		runex3 runex3_2 runex3_3\
		runex4 runex4_2 runex4_3 runex4_4 runex4_5 runex5\
		runex8 runex8_2 runex19
EXAMPLES_2    = ex10 ex11 ex6 ex8 ex13 ex14 ex15 ex18
RUNEXAMPLES_2 = runex11 runex6 runex6_2 runex6_3 runex6_4 runex6_5 runex6_6 \
		runex13 runex8_3 runex8_4
EXAMPLES_3    = ex7 ex12 ex16
RUNEXAMPLES_3 = runex7 runex16
EXAMPLES_4    = ex1 ex2 ex3 ex4 ex5 ex8 ex11 ex19
EXAMPLES_5    = ex17
RUNEXAMPLES_4 = runex1 runex1_3 runex1_5 runex2 runex2_3 runex2_5 \
		runex3 runex4 runex19
RUNEXAMPLES_5 = runex17
EXAMPLESC     = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex8.c ex9.c ex10.c\
		ex11.c ex13.c ex14.c ex15.c ex17.c ex18.c ex19.c
EXAMPLESF     = ex7.F ex12.F ex16.F

ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) ex1.o

ex2: ex2.o  chkopts
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) ex2.o

ex3: ex3.o  chkopts
	-$(CLINKER) -o ex3 ex3.o  $(PETSC_LIB)
	$(RM) ex3.o

ex4: ex4.o  chkopts
	-$(CLINKER) -o ex4 ex4.o  $(PETSC_LIB)
	$(RM) ex4.o

ex5: ex5.o  chkopts
	-$(CLINKER) -o ex5 ex5.o  $(PETSC_LIB)
	$(RM) ex5.o

ex6: ex6.o  chkopts
	-$(CLINKER) -o ex6 ex6.o  $(PETSC_LIB)
	$(RM) ex6.o

ex7: ex7.o  chkopts
	-$(FLINKER) -o ex7 ex7.o  $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
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

ex12: ex12.o  chkopts
	-$(FLINKER) -o ex12 ex12.o  $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex12.o

ex13: ex13.o  chkopts
	-$(CLINKER) -o ex13 ex13.o  $(PETSC_LIB)
	$(RM) ex13.o

ex14: ex14.o  chkopts
	-$(CLINKER) -o ex14 ex14.o  $(PETSC_LIB)
	$(RM) ex14.o

ex15: ex15.o  chkopts
	-$(CLINKER) -o ex15 ex15.o  $(PETSC_LIB)
	$(RM) ex15.o

ex16: ex16.o  chkopts
	-$(FLINKER) -o ex16 ex16.o  $(PETSC_FORTRAN_LIB) $(PETSC_LIB)
	$(RM) ex16.o

ex17: ex17.o  chkopts
	-$(CLINKER) -o ex17 ex17.o  $(PETSC_LIB)
	$(RM) ex17.o

ex18: ex18.o  chkopts
	-$(CLINKER) -o ex18 ex18.o  $(PETSC_LIB)
	$(RM) ex18.o

ex19: ex19.o  chkopts
	-$(CLINKER) -o ex19 ex19.o  $(PETSC_LIB)
	$(RM) ex19.o

runex1:
	-@$(MPIRUN) ex1 -ksp_smonitor > ex1_1.tmp 2>&1;	  \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   $(RM) -f ex1_1.tmp
runex1_2:
	-@$(MPIRUN) -np 2 ex1  -ksp_smonitor > ex1_2.tmp 2>&1;	 \
	   if (diff output/ex1_2.out ex1_2.tmp) then true; \
	   else echo "Possible problem with ex1_2, diffs above"; fi; \
	   $(RM) -f ex1_2.tmp
runex1_3:
	-@$(MPIRUN) ex1 -pc_type sor -pc_sor_symmetric -ksp_smonitor >\
	   ex1_3.tmp 2>&1;   \
	   if (diff output/ex1_3.out ex1_3.tmp) then true; \
	   else echo "Possible problem with ex1_3, diffs above"; fi; \
	   $(RM) -f ex1_3.tmp
runex1_4:
	-@$(MPIRUN) -np 2 ex1 -pc_type sor -pc_sor_symmetric -ksp_smonitor >\
	   ex1_4.tmp 2>&1;   \
	   if (diff output/ex1_4.out ex1_4.tmp) then true; \
	   else echo "Possible problem with ex1_4, diffs above"; fi; \
	   $(RM) -f ex1_4.tmp
runex1_5:
	-@$(MPIRUN) ex1 -pc_type eisenstat -ksp_smonitor >\
	   ex1_5.tmp 2>&1;   \
	   if (diff output/ex1_5.out ex1_5.tmp) then true; \
	   else echo "Possible problem with ex1_5, diffs above"; fi; \
	   $(RM) -f ex1_5.tmp
runex1_6:
	-@$(MPIRUN) -np 2 ex1 -pc_type eisenstat -ksp_smonitor >\
	   ex1_6.tmp 2>&1;   \
	   if (diff output/ex1_6.out ex1_6.tmp) then true; \
	   else echo "Possible problem with ex1_6, diffs above"; fi; \
	   $(RM) -f ex1_6.tmp
runex2:
	-@$(MPIRUN) ex2 -ksp_smonitor -m 5 > ex2_1.tmp 2>&1;   \
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   $(RM) -f ex2_1.tmp
runex2_2:
	-@$(MPIRUN) -np 2 ex2 -ksp_smonitor -m 5 > ex2_2.tmp 2>&1;   \
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
	-@$(MPIRUN) -np 2 ex2 -pc_type sor -pc_sor_symmetric -ksp_smonitor >\
	    ex2_4.tmp 2>&1;   \
	   if (diff output/ex2_4.out ex2_4.tmp) then true; \
	   else echo "Possible problem with ex2_4, diffs above"; fi; \
	   $(RM) -f ex2_4.tmp
runex2_5:
	-@$(MPIRUN) ex2 -pc_type eisenstat -ksp_smonitor >\
	    ex2_5.tmp 2>&1;   \
	   if (diff output/ex2_5.out ex2_5.tmp) then true; \
	   else echo "Possible problem with ex2_5, diffs above"; fi; \
	   $(RM) -f ex2_5.tmp
runex3:
	-@$(MPIRUN) ex3 -ksp_smonitor -m 5 > ex3_1.tmp 2>&1;   \
	   if (diff output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   $(RM) -f ex3_1.tmp
runex3_2:
	-@$(MPIRUN) -np 2 ex3 -ksp_smonitor -m 5 > ex3_2.tmp 2>&1;   \
	   if (diff output/ex3_2.out ex3_2.tmp) then true; \
	   else echo "Possible problem with ex3_2, diffs above"; fi; \
	   $(RM) -f ex3_2.tmp
runex3_3:
	-@$(MPIRUN) -np 2 ex3 -pc_type sor -pc_sor_symmetric -ksp_smonitor >\
	   ex3_3.tmp 2>&1;   \
	   if (diff output/ex3_3.out ex3_3.tmp) then true; \
	   else echo "Possible problem with ex3_3, diffs above"; fi; \
	   $(RM) -f ex3_3.tmp
runex4:
	-@$(MPIRUN) ex4 -ksp_smonitor -m 5 > ex4_1.tmp 2>&1;   \
	   if (diff output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   $(RM) -f ex4_1.tmp
runex4_2:
	-@$(MPIRUN) -np 2 ex4 -ksp_smonitor -m 5 > ex4_2.tmp 2>&1;   \
	   if (diff output/ex4_2.out ex4_2.tmp) then true; \
	   else echo "Possible problem with ex4_2, diffs above"; fi; \
	   $(RM) -f ex4_2.tmp
runex4_3:
	-@$(MPIRUN) ex4 -pc_type sor -pc_sor_symmetric -ksp_smonitor -m 5 >\
	   ex4_3.tmp 2>&1;   \
	   if (diff output/ex4_3.out ex4_3.tmp) then true; \
	   else echo "Possible problem with ex4_3, diffs above"; fi; \
	   $(RM) -f ex4_3.tmp
runex4_4:
	-@$(MPIRUN) -np 2 ex4 -pc_type sor -pc_sor_symmetric -ksp_smonitor -m 5 >\
	   ex4_4.tmp 2>&1;   \
	   if (diff output/ex4_4.out ex4_4.tmp) then true; \
	   else echo "Possible problem with ex4_4, diffs above"; fi; \
	   $(RM) -f ex4_4.tmp
runex4_5:
	-@$(MPIRUN) ex4 -pc_type eisenstat -ksp_smonitor -m 5	 >\
	   ex4_5.tmp 2>&1;   \
	   if (diff output/ex4_5.out ex4_5.tmp) then true; \
	   else echo "Possible problem with ex4_5, diffs above"; fi; \
	   $(RM) -f ex4_5.tmp
runex5:
	-@$(MPIRUN) -np 2 ex5 -ksp_smonitor -noslesview> ex5_1.tmp 2>&1; \
	   if (diff output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   $(RM) -f ex5_1.tmp
runex6:
	-@$(MPIRUN) ex6 -ksp_preonly  -pc_type lu \
	   -f ../../mat/examples/matrices/arco1 > ex6_1.tmp 2>&1; \
	   if (diff output/ex6_1.out ex6_1.tmp) then true; \
	   else echo "Possible problem with ex6_1, diffs above"; fi; \
	   $(RM) -f ex6_1.tmp
runex6_2:
	-@$(MPIRUN) ex6 -sub_pc_type ilu \
	   -f $(PETSC_DIR)/src/mat/examples/matrices/arco1 \
	   -ksp_gmres_restart 100 -sub_ksp_type preonly -pc_type bjacobi \
	   -pc_bjacobi_blocks 8 -sub_pc_ilu_in_place -ksp_smonitor \
	   > ex6_2.tmp 2>&1; \
	   if (diff output/ex6_2.out ex6_2.tmp) then true; \
	   else echo "Possible problem with ex6_2, diffs above"; fi; \
	   $(RM) -f ex6_2.tmp
runex6_3:
	-@$(MPIRUN) ex6 -pc_type ilu \
	   -f $(PETSC_DIR)/src/mat/examples/matrices/arco4 \
	   -ksp_gmres_restart 100  -ksp_smonitor  > ex6_3.tmp 2>&1; \
	   if (diff output/ex6_3.out ex6_3.tmp) then true; \
	   else echo "Possible problem with ex6_3, diffs above"; fi; \
	   $(RM) -f ex6_3.tmp
runex6_4:
	-@$(MPIRUN) ex6 -pc_type bgs -pc_bgs_blocks 5 \
	   -f $(PETSC_DIR)/src/mat/examples/matrices/arco1 \
	   -ksp_gmres_restart 100  -ksp_smonitor  > ex6_4.tmp 2>&1; \
	   if (diff output/ex6_4.out ex6_4.tmp) then true; \
	   else echo "Possible problem with ex6_4, diffs above"; fi; \
	   $(RM) -f ex6_4.tmp
runex6_5:
	-@$(MPIRUN) ex6 -pc_type bgs -pc_bgs_blocks 5 -mat_aij_oneindex \
	   -f $(PETSC_DIR)/src/mat/examples/matrices/arco1 \
	   -ksp_gmres_restart 100  -ksp_smonitor  > ex6_4.tmp 2>&1; \
	   if (diff output/ex6_4.out ex6_4.tmp) then true; \
	   else echo "Possible problem with ex6_5, diffs above"; fi; \
	   $(RM) -f ex6_4.tmp

# Note: We test these BlockSolve95 examples differently from the usual
# because the coloring heuristic is machine-dependent; convergence rates
# may differ slightly on various architectures.
runex6_6:
	-@$(MPIRUN) -np 3 ex6  -ksp_gmres_restart 100 \
	   -f $(PETSC_DIR)/src/mat/examples/matrices/arco1_ns \
	   -mat_mpirowbs -pc_type ilu -pc_ilu_factorpointwise 
	-@echo "This program tested the ILU preconditioner in BlockSolve95."
runex7:
	-@$(MPIRUN) -np 2 ex7 -ksp_smonitor > ex7_1.tmp 2>&1; \
	   if (diff output/ex7_1.out ex7_1.tmp) then true; \
	   else echo "Possible problem with ex7_1, diffs above"; fi; \
	   $(RM) -f ex7_1.tmp
runex8:
	-@$(MPIRUN) ex8 -ksp_smonitor > ex8_1.tmp 2>&1; \
	   if (diff output/ex8_1.out ex8_1.tmp) then true; \
	   else echo "Possible problem with ex8_1, diffs above"; fi; \
	   $(RM) -f ex8_1.tmp
runex8_2:
	-@$(MPIRUN) -np 2 ex8 -ksp_smonitor -ksp_rtol .000001 >\
	   ex8_2.tmp 2>&1;   \
	   if (diff output/ex8_2.out ex8_2.tmp) then true; \
	   else echo "Possible problem with ex8_2, diffs above"; fi; \
	   $(RM) -f ex8_2.tmp

# Note: We test these BlockSolve95 examples differently from the usual
# because the coloring heuristic is machine-dependent; convergence rates
# may differ slightly on various architectures.
runex8_3:
	-@$(MPIRUN) -np 2 ex8 -pc_type icc -mat_mpirowbs -ksp_rtol .000001;\
	-@echo "This program tested the ICC preconditioner in BlockSolve95."
runex8_4:
	-@$(MPIRUN) -np 2 ex8 -pc_type ilu -mat_nonsym -mat_mpirowbs \
	   -ksp_rtol .000001;\
	-@echo "This program tested the ILU preconditioner in BlockSolve95."
runex10:
	-@$(MPIRUN) ex10 -mat_bdiag -ksp_smonitor > ex10_1.tmp 2>&1; \
	   if (diff output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Possible problem with ex10_1, diffs above"; fi; \
	   $(RM) -f ex10_1.tmp
runex11:
	-@$(MPIRUN) -np 3  ex11 -ksp_smonitor > ex11_1.tmp 2>&1; \
	   if (diff output/ex11_1.out ex11_1.tmp) then true; \
	   else echo "Possible problem with ex11_1, diffs above"; fi; \
	   $(RM) -f ex11_1.tmp
runex13:
	-@$(MPIRUN) -np 2  ex13 -ksp_smonitor > ex13_1.tmp 2>&1; \
	   if (diff output/ex13_1.out ex13_1.tmp) then true; \
	   else echo "Possible problem with ex13_1, diffs above"; fi; \
	   $(RM) -f ex13_1.tmp
runex16:
	-@$(MPIRUN) ex16 -f $(PETSC_DIR)/src/mat/examples/matbinary.ex \
	   -a_pc_type bgs -a_pc_bgs_blocks 5 -a_sub_pc_type ilu \
	   -b_pc_type bjacobi -b_pc_bjacobi_blocks 5 -b_ksp_max_it 10 \
	   -b_sub_pc_type bgs -b_sub_pc_bgs_blocks 2  > ex16_1.tmp 2>&1; \
	   if (diff output/ex16_1.out ex16_1.tmp) then true; \
	   else echo "Possible problem with ex16_1, diffs above"; fi; \
	   $(RM) -f ex13_1.tmp

runex17:
	-@$(MPIRUN) ex17 -n 20 -ksp_smonitor -p 4 -ksp_type cg \
	   -ksp_cg_symmetric  > ex17_1.tmp 2>&1; \
	   if (diff output/ex17_1.out ex17_1.tmp) then true; \
	   else echo "Possible problem with ex17_1, diffs above"; fi; \
	   $(RM) -f ex17_1.tmp

runex19:
	-@$(MPIRUN) ex19 -t 2 -ksp_smonitor -ksp_type gmres \
	   -s2_ksp_type bcgs -s2_ksp_smonitor > ex19_1.tmp 2>&1; \
	   if (diff output/ex19_1.out ex19_1.tmp) then true; \
	   else echo "Possible problem with ex19_1, diffs above"; fi; \
	   $(RM) -f ex19_1.tmp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
