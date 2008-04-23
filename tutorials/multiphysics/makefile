
CFLAGS           =
FFLAGS		 = -I${PETSC_DIR}/include/finclude
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = tutorials/multiphysics
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
runp1:
	-@${MPIEXEC} -n 1 ./p1 -snes_monitor_short > p1_1.tmp 2>&1;   \
	   if (${DIFF} output/p1_1.out p1_1.tmp) then true; \
	   else echo "Possible problem with p1_1, diffs above"; fi; \
	   ${RM} -f p1_1.tmp

runp2:
	-@${MPIEXEC} -n 1 ./p2 -snes_monitor_short > p2_1.tmp 2>&1;   \
	   if (${DIFF} output/p2_1.out p2_1.tmp) then true; \
	   else echo "Possible problem with p2_1, diffs above"; fi; \
	   ${RM} -f p2_1.tmp

runmp:
	-@${MPIEXEC} -n 1 ./mp -snes_monitor_short > mp_1.tmp 2>&1;   \
	   if (${DIFF} output/mp_1.out mp_1.tmp) then true; \
	   else echo "Possible problem with mp_1, diffs above"; fi; \
	   ${RM} -f mp_1.tmp

runmp_2:
	-@${MPIEXEC} -n 1 ./mp -dmmg_jacobian_mf_fd_operator > mp_2.tmp 2>&1;   \
	   if (${DIFF} output/mp_2.out mp_2.tmp) then true; \
	   else echo "Possible problem with mp_2, diffs above"; fi; \
	   ${RM} -f mp_2.tmp

runmp_3:
	-@${MPIEXEC} -n 1 ./mp -dmmg_jacobian_mf_fd_operator -dmcomposite_dense_jacobian > mp_3.tmp 2>&1;   \
	   if (${DIFF} output/mp_3.out mp_3.tmp) then true; \
	   else echo "Possible problem with mp_3, diffs above"; fi; \
	   ${RM} -f mp_3.tmp

TESTEXAMPLES_C		       = p1.PETSc runp1 p1.rm p2.PETSc runp2 p2.rm mp.PETSc runmp runmp_2 runmp_3 mp.rm mp.rm 
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
