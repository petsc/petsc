# $Id: makefile,v 1.12 1998/04/29 02:50:51 bsmith Exp bsmith $ 

ALL: ex1

CFLAGS 	        = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
FFLAGS	        = 
LOCDIR          = src/mat/examples/tutorials/

TESTEXAMPLES_1  = 
TESTEXAMPLES_2  =
TESTEXAMPLES_3  =
TESTEXAMPLES_4  =
TESTEXAMPLES_6  = ex2.PETSc runex2 runex2_2 ex2.rm
TESTEXAMPLES_7  =
TESTEXAMPLES_9  =
BUILDEXAMPLES_1 =
BUILDEXAMPLES_2 =
BUILDEXAMPLES_3 =
BUILDEXAMPLES_4 =
BUILDEXAMPLES_6 =
BUILDEXAMPLES_7 =
BUILDEXAMPLES_9 =

EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       =

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_LIB} 
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_LIB} 
	${RM} ex2.o

runex2:
	-@${MPIRUN} -np 1 ex2  >\
	   ex2_1.tmp 2>&1;   \
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp

runex2_2:
	-@${MPIRUN} -np 2 ex2  >\
	   ex2_2.tmp 2>&1;   \
	   if (diff output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi; \
	   ${RM} -f ex2_2.tmp


