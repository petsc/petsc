IPETSCDIR = ../../../
ALL: ex1

CFLAGS    = $(OPT) $(CONF) $(INCLUDEDIRS) $(BS_INCLUDE)
SOURCEC   = 
SOURCEF   =
WSOURCEC  = 
SOURCEH   = pargrid.h
OBJSC     =
WOBJS     = 
OBJSF     =
LIBBASE   = libpetscsles
EXAMPLES  = ex1 ex2 ex3 ex4 ex8
REXAMPLES = runex1 runex1_2 runex1_3 runex1_4 runex1_5 runex1_6\
            runex2 runex2_2 runex2_3 runex2_4 runex2_5\
            runex3 runex3_2 runex3_3\
            runex4 runex4_2 runex4_3 runex4_4 runex4_5 runex8 runex8_2
CEXAMPLES = ex1 ex2 ex3 ex4 ex8 ex9 ex10
EXAMPLESC = ex1.c ex2.c ex3.c ex4.c ex8.c ex9.c ex10.c
LINCLUDE  = $(SOURCEH)

runex1:
	-@$(MPIRUN) ex1 -ksp_monitor > ex1_1.tmp 2>&1;   \
           if (diff ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           $(RM) -f ex1_1.tmp
runex1_2:
	-@$(MPIRUN) -np 2 ex1  -ksp_monitor > ex1_2.tmp 2>&1;   \
           if (diff ex1_2.out ex1_2.tmp) then true; \
           else echo "Possible problem with ex1_2, diffs above"; fi; \
           $(RM) -f ex1_2.tmp
runex1_3:
	-@$(MPIRUN) -np 1 ex1 -pc_method sor -pc_sor_symmetric -ksp_monitor >\
           ex1_3.tmp 2>&1;   \
           if (diff ex1_3.out ex1_3.tmp) then true; \
           else echo "Possible problem with ex1_3, diffs above"; fi; \
           $(RM) -f ex1_3.tmp
runex1_4:
	-@$(MPIRUN) -np 2 ex1 -pc_method sor -pc_sor_symmetric -ksp_monitor >\
           ex1_4.tmp 2>&1;   \
           if (diff ex1_4.out ex1_4.tmp) then true; \
           else echo "Possible problem with ex1_4, diffs above"; fi; \
           $(RM) -f ex1_4.tmp
runex1_5:
	-@$(MPIRUN) -np 1 ex1 -pc_method eisenstat -ksp_monitor >\
           ex1_5.tmp 2>&1;   \
           if (diff ex1_5.out ex1_5.tmp) then true; \
           else echo "Possible problem with ex1_5, diffs above"; fi; \
           $(RM) -f ex1_5.tmp
runex1_6:
	-@$(MPIRUN) -np 2 ex1 -pc_method eisenstat -ksp_monitor >\
           ex1_6.tmp 2>&1;   \
           if (diff ex1_6.out ex1_6.tmp) then true; \
           else echo "Possible problem with ex1_6, diffs above"; fi; \
           $(RM) -f ex1_6.tmp
runex2:
	-@$(MPIRUN) ex2 -ksp_monitor -m 5 > ex2_1.tmp 2>&1;   \
           if (diff ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi; \
           $(RM) -f ex2_1.tmp
runex2_2:
	-@$(MPIRUN) -np 2 ex2 -ksp_monitor -m 5 > ex2_2.tmp 2>&1;   \
           if (diff ex2_2.out ex2_2.tmp) then true; \
           else echo "Possible problem with ex2_2, diffs above"; fi; \
           $(RM) -f ex2_2.tmp
runex2_3:
	-@$(MPIRUN) -np 1 ex2 -pc_method sor -pc_sor_symmetric -ksp_monitor > \
            ex2_3.tmp 2>&1;   \
           if (diff ex2_3.out ex2_3.tmp) then true; \
           else echo "Possible problem with ex2_3, diffs above"; fi; \
           $(RM) -f ex2_3.tmp
runex2_4:
	-@$(MPIRUN) -np 2 ex2 -pc_method sor -pc_sor_symmetric -ksp_monitor >\
            ex2_4.tmp 2>&1;   \
           if (diff ex2_4.out ex2_4.tmp) then true; \
           else echo "Possible problem with ex2_4, diffs above"; fi; \
           $(RM) -f ex2_4.tmp
runex2_5:
	-@$(MPIRUN) -np 1 ex2 -pc_method eisenstat -ksp_monitor >\
            ex2_5.tmp 2>&1;   \
           if (diff ex2_5.out ex2_5.tmp) then true; \
           else echo "Possible problem with ex2_5, diffs above"; fi; \
           $(RM) -f ex2_5.tmp
runex3:
	-@$(MPIRUN) ex3 -ksp_monitor -m 5 > ex3_1.tmp 2>&1;   \
           if (diff ex3_1.out ex3_1.tmp) then true; \
           else echo "Possible problem with ex3_1, diffs above"; fi; \
           $(RM) -f ex3_1.tmp
runex3_2:
	-@$(MPIRUN) -np 2 ex3 -ksp_monitor -m 5 > ex3_2.tmp 2>&1;   \
           if (diff ex3_2.out ex3_2.tmp) then true; \
           else echo "Possible problem with ex3_2, diffs above"; fi; \
           $(RM) -f ex3_2.tmp
runex3_3:
	-@$(MPIRUN) -np 2 ex3 -pc_method sor -pc_sor_symmetric -ksp_monitor >\
           ex3_3.tmp 2>&1;   \
           if (diff ex3_3.out ex3_3.tmp) then true; \
           else echo "Possible problem with ex3_3, diffs above"; fi; \
           $(RM) -f ex3_3.tmp
runex4:
	-@$(MPIRUN) ex4 -ksp_monitor -m 5 > ex4_1.tmp 2>&1;   \
           if (diff ex4_1.out ex4_1.tmp) then true; \
           else echo "Possible problem with ex4_1, diffs above"; fi; \
           $(RM) -f ex4_1.tmp
runex4_2:
	-@$(MPIRUN) -np 2 ex4 -ksp_monitor -m 5 > ex4_2.tmp 2>&1;   \
           if (diff ex4_2.out ex4_2.tmp) then true; \
           else echo "Possible problem with ex4_2, diffs above"; fi; \
           $(RM) -f ex4_2.tmp
runex4_3:
	-@$(MPIRUN) -np 1 ex4 -pc_method sor -pc_sor_symmetric -ksp_monitor -m 5 >\
           ex4_3.tmp 2>&1;   \
           if (diff ex4_3.out ex4_3.tmp) then true; \
           else echo "Possible problem with ex4_3, diffs above"; fi; \
           $(RM) -f ex4_3.tmp
runex4_4:
	-@$(MPIRUN) -np 2 ex4 -pc_method sor -pc_sor_symmetric -ksp_monitor -m 5 >\
           ex4_4.tmp 2>&1;   \
           if (diff ex4_4.out ex4_4.tmp) then true; \
           else echo "Possible problem with ex4_4, diffs above"; fi; \
           $(RM) -f ex4_4.tmp
runex4_5:
	-@$(MPIRUN) -np 1 ex4 -pc_method eisenstat -ksp_monitor -m 5  >\
           ex4_5.tmp 2>&1;   \
           if (diff ex4_5.out ex4_5.tmp) then true; \
           else echo "Possible problem with ex4_5, diffs above"; fi; \
           $(RM) -f ex4_5.tmp
runex8:
	-@$(MPIRUN) ex8 -ksp_monitor > ex8_1.tmp 2>&1; \
           if (diff ex8_1.out ex8_1.tmp) then true; \
           else echo "Possible problem with ex8_1, diffs above"; fi; \
           $(RM) -f ex8_1.tmp
runex8_2:
	-@$(MPIRUN) -np 2 ex8 -ksp_monitor -ksp_rtol .000001 >\
           ex8_2.tmp 2>&1;   \
           if (diff ex8_2.out ex8_2.tmp) then true; \
           else echo "Possible problem with ex8_2, diffs above"; fi; \
           $(RM) -f ex8_2.tmp

ex1: ex1.o 
	-$(CLINKER) -o ex1 ex1.o  $(PETSC_LIB)
	$(RM) ex1.o

ex2: ex2.o 
	-$(CLINKER) -o ex2 ex2.o  $(PETSC_LIB)
	$(RM) ex2.o

ex3: ex3.o 
	-$(CLINKER) -o ex3 ex3.o  $(PETSC_LIB)
	$(RM) ex3.o

ex4: ex4.o 
	-$(CLINKER) -o ex4 ex4.o  $(PETSC_LIB)
	$(RM) ex4.o

ex5: ex5.o 
	-$(CLINKER) -o ex5 ex5.o  $(PETSC_LIB)
	$(RM) ex5.o

ex6: ex6.o 
	-$(CLINKER) -o ex6 ex6.o  $(PETSC_LIB)
	$(RM) ex6.o

ex7: ex7.o 
	-$(CLINKER) -o ex7 ex7.o  $(PETSC_LIB)
	$(RM) ex7.o

ex8: ex8.o 
	-$(CLINKER) -o ex8 ex8.o  $(PETSC_LIB)
	$(RM) ex8.o

ex9: ex9.o 
	-$(CLINKER) -o ex9 ex9.o  $(PETSC_LIB)
	$(RM) ex9.o

ex10: ex10.o 
	-$(CLINKER) -o ex10 ex10.o  $(PETSC_LIB)
	$(RM) ex10.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
