# $Id: makefile,v 1.9 1998/06/12 22:19:49 balay Exp bsmith $ 

ALL: ex1

CFLAGS 	        =
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/adic/examples/tutorials/

TESTEXAMPLES_C		    = 
TESTEXAMPLES_C_X11	    = 
TESTEXAMPLES_FORTRAN	    = 
TESTEXAMPLES_C_X11_MPIUNI   = 
TESTEXAMPLES_C_NOCOMPLEX    = 
TESTEXAMPLES_FORTRAN_MPIUNI = 

EXAMPLESC       = ex1.c ex2.c ex3.c ex3func.c ex4.c ex5.c
EXAMPLESF       =

include ${PETSC_DIR}/bmake/common/base

ex1.ad:  chkopts 
	${ADIC_CC} -h ${CFLAGS} ex1.c
	${CC} ${ADIC_INCLUDE} ${CFLAGS} ${COPTFLAGS} -c ex1.ad.c
	-${CLINKER}  -o ex1.ad ex1.ad.o  ${PETSC_ADIC_LIB} ${ADIC_LIB}
	${RM} ex1.o

ex2:  chkopts ex2.c ex2func.c
	${CC} ${ADIC_INCLUDE} ${CFLAGS} ${COPTFLAGS} -c ex2.c
	${ADIC_CC} -h ${CFLAGS} ex2func.c
	${CC} ${ADIC_INCLUDE} ${CFLAGS} ${COPTFLAGS} -c ex2func.ad.c 
	-${CLINKER}  -o ex2 ex2.o ex2func.o ex2func.ad.o ${PETSC_ADIC_LIB} ${ADIC_LIB}
	${RM} ex2.o ex2func.o ex2func.ad.o

ex3:  chkopts ex3.o ex3func.o
	${ADIC_CC} -h ${CFLAGS} ex3func.c
	${CC} ${ADIC_INCLUDE} ${CFLAGS} ${COPTFLAGS} -c ex3func.ad.c 
	-${CLINKER}  -o ex3 ex3.o ex3func.o ex3func.ad.o ${PETSC_ADIC_LIB} ${ADIC_LIB}
	${RM} ex3.o ex3func.o ex3func.ad.o

ex4:  chkopts ex4.o ex4func.o
	${ADIC_CC} -h ${CFLAGS} ex4func.c
	${CC} ${ADIC_INCLUDE} ${CFLAGS} ${COPTFLAGS} -c ex4func.ad.c 
	-${CLINKER}  -o ex4 ex4.o ex4func.o ex4func.ad.o ${PETSC_ADIC_LIB} ${ADIC_LIB}
	${RM} ex4.o ex4func.o ex4func.ad.o

ex4tmp:  chkopts ex4tmp.o ex4tmpfunc.o 
	${ADIC_CC} -h ${CFLAGS} ex4tmpfunc.c
	${CC} ${ADIC_INCLUDE} ${CFLAGS} ${COPTFLAGS} -c ex4tmpfunc.ad.c 
	${CC} ${ADIC_INCLUDE} ${CFLAGS} ${COPTFLAGS} -c adstuff.c
	-${CLINKER}  -o ex4tmp ex4tmp.o ex4tmpfunc.o ex4tmpfunc.ad.o adstuff.o ${PETSC_ADIC_LIB} ${ADIC_LIB}
	${RM} ex4tmp.o ex4tmpfunc.o ex4tmpfunc.ad.o addstuff.o

ex5:  chkopts ex5.o ex5func.o
	${ADIC_CC} -h ${CFLAGS} ex5func.c
	${CC} ${ADIC_INCLUDE} ${CFLAGS} ${COPTFLAGS} -c ex5func.ad.c 
	-${CLINKER}  -o ex5 ex5.o ex5func.o ex5func.ad.o ${PETSC_ADIC_LIB} ${ADIC_LIB}
	${RM} ex5.o ex5func.o ex5func.ad.o


