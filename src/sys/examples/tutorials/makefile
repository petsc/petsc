# $Id: makefile,v 1.18 1998/04/29 00:46:22 balay Exp balay $ 

ALL:

CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/examples/tutorials

TESTEXAMPLES_1  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm \
                   ex3.PETSc runex3 ex3.rm clean
TESTEXAMPLES_4  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm \
                   ex3.PETSc runex3 ex3.rm clean
TESTEXAMPLES_3  = ex3f.PETSc runex3f ex3f.rm clean
BUILDEXAMPLES_1 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm \
                  ex3.PETSc ex3.rm clean
BUILDEXAMPLES_4 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm clean
BUILDEXAMPLES_3 = ex3f.PETSc ex3f.rm clean

EXAMPLESC       = ex1.c ex2.c ex3.c
EXAMPLESF       = ex3f.F

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o 
	-${CLINKER} -o ex1 ex1.o  ${PETSC_LIB}
	${RM} -f ex1.o

ex2: ex2.o 
	-${CLINKER} -o ex2 ex2.o  ${PETSC_LIB}
	${RM} -f ex2.o

ex3: ex3.o 
	-${CLINKER} -o ex3 ex3.o  ${PETSC_LIB}
	${RM} -f ex3.o

ex3f: ex3f.o 
	-${FLINKER} -o ex3f ex3f.o  ${PETSC_FORTRAN_LIB} ${PETSC_LIB}
	${RM} -f ex3f.o

#
# ------------------------------------------------------------------------
#
runex1:
	-@${MPIRUN} -np 1 ex1 > ex1_1.tmp 2>&1; \
	   if (diff output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp 

runex2:
	-@${MPIRUN} -np 1 ex2 > ex2_1.tmp 2>&1; \
	   if (diff output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp 

runex3:
	-@${MPIRUN} -np 1 ex3 

runex3f:
	-@${MPIRUN} -np 1 ex3f



