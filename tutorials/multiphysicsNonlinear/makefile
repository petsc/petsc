
CFLAGS           =
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = tutorials/multiphysicsNonlinear/
MANSEC           = SNES
EXAMPLESC	 = p1.c p2.c ff1.c ff2.c mp1.c
EXAMPLESF	 = 
EXAMPLESCH	 = mp1.h
EXAMPLESFH       = 
EXAMPLESMATLAB   = 
DIRS		 = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

p1: p1.o  ff1.o ff2.o chkopts
	-${CLINKER} -o p1 p1.o ff1.o ff2.o ${PETSC_SNES_LIB}
	${RM} p1.o ff1.o ff2.o

p2: p2.o  ff1.o ff2.o chkopts
	-${CLINKER} -o p2 p2.o ff1.o ff2.o ${PETSC_SNES_LIB}
	${RM} p2.o ff1.o ff2.o

mp1: mp1.o  ff1.o ff2.o chkopts
	-${CLINKER} -o mp1 mp1.o ff1.o ff2.o ${PETSC_SNES_LIB}
	${RM} mp1.o ff1.o ff2.o
#--------------------------------------------------------------------------


TESTEXAMPLES_C		       = 
TESTEXAMPLES_C_X11	       = 
TESTEXAMPLES_FORTRAN	       = 
TESTEXAMPLES_C_NOCOMPLEX       = 
TESTEXAMPLES_FORTRAN_NOCOMPLEX = 
TESTEXAMPLES_FORTRAN_MPIUNI    = 
TESTEXAMPLES_C_X11_MPIUNI      = 
TESTEXAMPLES_F90	       = 
TESTEXAMPLES_13		       = 
TESTEXAMPLES_MATLAB_ENGINE     =
TESTEXAMPLES_AMS	       = 
TESTEXAMPLES_ADIFOR	       =
TESTEXAMPLES_MUMPS             =

include ${PETSC_DIR}/conf/test
