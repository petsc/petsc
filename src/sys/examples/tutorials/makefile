# $Id: makefile,v 1.13 1998/04/23 22:48:53 balay Exp bsmith $ 

ALL:

CFLAGS	        = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
FFLAGS	        = 
SOURCEC	        = 
SOURCEF	        =
SOURCEH	        =
OBJSC	        =
OBJSF	        =
LIBBASE	        = libpetscsys
LOCDIR          = src/sys/examples/tutorials
TESTEXAMPLES_1  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm clean
TESTEXAMPLES_4  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm clean
BUILDEXAMPLES_1 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm clean
BUILDEXAMPLES_4 = ex1.PETSc ex1.rm ex2.PETSc ex2.rm clean

EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

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


ex1: ex1.o 
	-${CLINKER} -o ex1 ex1.o  ${PETSC_LIB}
	${RM} -f ex1.o

ex2: ex2.o 
	-${CLINKER} -o ex2 ex2.o  ${PETSC_LIB}
	${RM} -f ex2.o

ex3: ex3.o 
	-${CLINKER} -o ex3 ex3.o  ${PETSC_LIB}
	${RM} -f ex3.o

ex4: ex4.o 
	-${CLINKER} -o ex4 ex4.o  ${PETSC_LIB}
	${RM} -f ex4.o

ex5: ex5.o 
	-${FLINKER} -o ex5 ex5.o  ${PETSC_FORTRAN_LIB} ${PETSC_LIB}
	${RM} -f ex5.o

ex6: ex6.o 
	-${FLINKER} -o ex6 ex6.o  ${PETSC_FORTRAN_LIB} ${PETSC_LIB}
	${RM} -f ex6.o

