IPETSCDIR = ../../../
ALL: ex1

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(COPT) $(CONF) -I$(IPETSCDIR)/include
SOURCEC  = 
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
EXAMPLES = ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10 ex11 ex12 ex13 ex14 \
           ex16 ex17
REXAMPLES= runex1 runex1_2 runex2 runex3 runex4 runex5 runex6 runex7 runex8\
           runex9 runex10 runex11 runex12 runex13 runex14 runex16 runex17
CEXAMPLES= ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10 ex11 ex12 ex13 ex14 \
           ex15 ex16 ex17 ex18
EXAMPLESC= ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c \
           ex11.c ex12.c ex13.c ex14.c ex15.c ex16.c ex17.c ex18.c
LINCLUDE = $(SOURCEH)

runex1:
	-@$(MPIRUN)  ex1 > ex1_1.tmp 2>&1;   \
           if (diff output/ex1_1.out ex1_1.tmp) then true; \
           else echo "Possible problem with ex1_1, diffs above"; fi; \
           $(RM) -f ex1_1.tmp
runex1_2:
	-@$(MPIRUN)  -np 2 ex1 > ex1_2.tmp 2>&1;\
           if (diff output/ex1_2.out ex1_2.tmp) then true ;  \
           else echo "Possible problem with ex1_2, diffs above"; fi ;\
           $(RM) -f ex1_2.tmp
runex2:
	-@$(MPIRUN)  ex2 > ex2_1.tmp 2>&1;\
           if (diff output/ex2_1.out ex2_1.tmp) then true; \
           else echo "Possible problem with ex2_1, diffs above"; fi;\
           $(RM) -f ex2_1.tmp
runex3:
	-@$(MPIRUN) -np 2 ex3 > ex3_1.tmp 2>&1;\
           if (diff output/ex3_1.out ex3_1.tmp) then true; \
           else echo "Possible problem with ex3_1, diffs above"; fi;\
           $(RM) -f ex3_1.tmp
runex4:
	-@$(MPIRUN) -np 2 ex4 > ex4_1.tmp 2>&1;\
           if (diff output/ex4_1.out ex4_1.tmp) then true; \
           else echo "Possible problem with ex4_1, diffs above"; fi;\
           $(RM) -f ex4_1.tmp
runex5:
	-@$(MPIRUN) -np 2 ex5 > ex5_1.tmp 2>&1;\
           if (diff output/ex5_1.out ex5_1.tmp) then true; \
           else echo "Possible problem with ex5_1, diffs above"; fi;\
           $(RM) -f ex5_1.tmp
runex6:
	-@$(MPIRUN)  ex6 > ex6_1.tmp 2>&1;\
           if (diff output/ex6_1.out ex6_1.tmp) then true; \
           else echo "Possible problem with ex6_1, diffs above"; fi;\
           $(RM) -f ex6_1.tmp
runex7:
	-@$(MPIRUN)  ex7 > ex7_1.tmp 2>&1;\
           if (diff output/ex7_1.out ex7_1.tmp) then true; \
           else echo "Possible problem with ex7_1, diffs above"; fi;\
           $(RM) -f ex7_1.tmp
runex8:
	-@$(MPIRUN)  ex8 > ex8_1.tmp 2>&1;\
           if (diff output/ex8_1.out ex8_1.tmp) then true; \
           else echo "Possible problem with ex8_1, diffs above"; fi;\
           $(RM) -f ex8_1.tmp
runex9:
	-@$(MPIRUN)  -np 2 ex9 > ex9_1.tmp 2>&1;\
           if (diff output/ex9_1.out ex9_1.tmp) then true; \
           else echo "Possible problem with ex9_1, diffs above"; fi;\
           $(RM) -f ex9_1.tmp
runex10:
	-@$(MPIRUN)  -np 2 ex10 > ex10_1.tmp 2>&1;\
           if (diff output/ex10_1.out ex10_1.tmp) then true; \
           else echo "Possible problem with ex10_1, diffs above"; fi;\
           $(RM) -f ex10_1.tmp
runex11:
	-@$(MPIRUN)  -np 2 ex11 > ex11_1.tmp 2>&1;\
           if (diff output/ex11_1.out ex11_1.tmp) then true; \
           else echo "Possible problem with ex11_1, diffs above"; fi;\
           $(RM) -f ex11_1.tmp
runex12:
	-@$(MPIRUN)  -np 2 ex12 > ex12_1.tmp 2>&1;\
           if (diff output/ex12_1.out ex12_1.tmp) then true; \
           else echo "Possible problem with ex12_1, diffs above"; fi;\
           $(RM) -f ex12_1.tmp
runex13:
	-@$(MPIRUN)  -np 2 ex13 > ex13_1.tmp 2>&1;\
           if (diff output/ex13_1.out ex13_1.tmp) then true; \
           else echo "Possible problem with ex13_1, diffs above"; fi;\
           $(RM) -f ex13_1.tmp
runex14:
	-@$(MPIRUN)  -np 2 ex14 > ex14_1.tmp 2>&1;\
           if (diff output/ex14_1.out ex14_1.tmp) then true; \
           else echo "Possible problem with ex14_1, diffs above"; fi;\
           $(RM) -f ex14_1.tmp
runex16:
	-@$(MPIRUN) -np 3 ex16 > ex16_1.tmp 2>&1;\
           if (diff output/ex16_1.out ex16_1.tmp) then true; \
           else echo "Possible problem with ex16_1, diffs above"; fi;\
           $(RM) -f ex16_1.tmp
runex17:
	-@$(MPIRUN)  -np 2 ex17 > ex17_1.tmp 2>&1;\
           if (diff output/ex17_1.out ex17_1.tmp) then true; \
           else echo "Possible problem with ex17_1, diffs above"; fi;\
           $(RM) -f ex17_1.tmp
	
ex1: ex1.o  chkopts
	-$(CLINKER) -o ex1 ex1.o  $(SLES_LIB)
	$(RM) -f ex1.o

ex2: ex2.o  chkopts
	-$(CLINKER) -o ex2 ex2.o  $(SLES_LIB)
	$(RM) -f ex2.o

ex3: ex3.o  chkopts
	-$(CLINKER) -o ex3 ex3.o  $(SLES_LIB)
	$(RM) -f ex3.o

ex4: ex4.o  chkopts
	-$(CLINKER) -o ex4 ex4.o  $(SLES_LIB)
	$(RM) -f ex4.o

ex5: ex5.o  chkopts
	-$(CLINKER) -o ex5 ex5.o  $(SLES_LIB)
	$(RM) -f ex5.o

ex6: ex6.o  chkopts
	-$(CLINKER) -o ex6 ex6.o  $(SLES_LIB)
	$(RM) -f ex6.o

ex7: ex7.o  chkopts
	-$(CLINKER) -o ex7 ex7.o  $(SLES_LIB)
	$(RM) -f ex7.o

ex8: ex8.o  chkopts
	-$(CLINKER) -o ex8 ex8.o  $(SLES_LIB)
	$(RM) -f ex8.o

ex9: ex9.o  chkopts
	-$(CLINKER) -o ex9 ex9.o  $(SLES_LIB)
	$(RM) -f ex9.o

ex10: ex10.o  chkopts
	-$(CLINKER) -o ex10 ex10.o  $(SLES_LIB)
	$(RM) -f ex10.o

ex11: ex11.o  chkopts
	-$(CLINKER) -o ex11 ex11.o  $(SLES_LIB)
	$(RM) -f ex11.o

ex12: ex12.o  chkopts
	-$(CLINKER) -o ex12 ex12.o  $(SLES_LIB)
	$(RM) -f ex12.o

ex13: ex13.o  chkopts
	-$(CLINKER) -o ex13 ex13.o  $(SLES_LIB)
	$(RM) -f ex13.o

ex14: ex14.o  chkopts
	-$(CLINKER) -o ex14 ex14.o  $(SLES_LIB)
	$(RM) -f ex14.o

ex15: ex15.o  chkopts
	-$(CLINKER) -o ex15 ex15.o  $(SLES_LIB) 
	$(RM) -f ex15.o

ex16: ex16.o  chkopts
	-$(CLINKER) -o ex16 ex16.o  $(SLES_LIB) 
	$(RM) -f ex16.o

ex17: ex17.o  chkopts
	-$(CLINKER) -o ex17 ex17.o  $(SLES_LIB)
	$(RM) -f ex17.o

ex18: ex18.o  chkopts
	-$(CLINKER) -o ex18 ex18.o  $(SLES_LIB)
	$(RM) -f ex18.o

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

