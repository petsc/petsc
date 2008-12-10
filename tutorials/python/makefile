
CFLAGS           =
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = tutorials/python
MANSEC           = 
EXAMPLESC	 = ex1.c
EXAMPLESF	 = 
EXAMPLESCH	 = 
EXAMPLESFH       = 
EXAMPLESMATLAB   = 
DIRS		 = 
CLEANFILES       = 

include ${PETSC_DIR}/conf/base

#--------------------------------------------------------------------------

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_KSP_LIB}
	${RM} ex1.o

#--------------------------------------------------------------------------

PYTHON = python

runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ksp_monitor_short -ksp_view > ex1_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp example1.py[co]

runex1py:
	-@${MPIEXEC} -n 1 python ./ex1.py -ksp_monitor_short -ksp_view > ex1py_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1py_1.out ex1py_1.tmp) then true; \
	   else echo "Possible problem with ex1py_1, diffs above"; fi; \
	   ${RM} -f ex1py_1.tmp example1.py[co]

TESTEXAMPLES_C = ex1.PETSc runex1 runex1py ex1.rm

#--------------------------------------------------------------------------

include ${PETSC_DIR}/conf/test
