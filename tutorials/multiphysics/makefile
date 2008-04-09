
CFLAGS           =
FFLAGS		 = -I${PETSC_DIR}/include/finclude
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = tutorials/multi-physics
MANSEC           = SNES
EXAMPLESC	 = mp.c p1.c p2.c ff1.c ff2.c
EXAMPLESF	 = 
EXAMPLESCH	 = mp.h
EXAMPLESFH       = 
EXAMPLESMATLAB   = 
DIRS		 = 

include ${PETSC_DIR}/conf/base

p1: p1.o  ff1.o ff2.o chkopts
	-${CLINKER} -o p1 p1.o ff1.o ff2.o ${PETSC_SNES_LIB}
	${RM} p1.o ff1.o ff2.o

p2: p2.o  ff1.o ff2.o chkopts
	-${CLINKER} -o p2 p2.o ff1.o ff2.o ${PETSC_SNES_LIB}
	${RM} p2.o ff1.o ff2.o

mp: mp.o  ff1.o ff2.o chkopts
	-${CLINKER} -o mp mp.o ff1.o ff2.o ${PETSC_SNES_LIB}
	${RM} mp.o ff1.o ff2.o
#--------------------------------------------------------------------------
runmp:
	-@${MPIEXEC} -np 1 ./mp > mp_1.tmp 2>&1;   \
	   if (${DIFF} output/mp_1.out mp_1.tmp) then true; \
	   else echo "Possible problem with mp_1, diffs above"; fi; \
	   ${RM} -f mp_1.tmp

TESTEXAMPLES_C		       = mp.PETSc runmp mp.rm 
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
