
CFLAGS           =
FFLAGS		 = -I${PETSC_DIR}/include/finclude
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = tutorials/multi-physics
MANSEC           = SNES
EXAMPLESC	 = mp1.c
EXAMPLESF	 = 
EXAMPLESCH	 = 
EXAMPLESFH       = 
EXAMPLESMATLAB   = 
DIRS		 = 

include ${PETSC_DIR}/conf/base

mp1: mp1.o  chkopts
	-${CLINKER} -o mp1 mp1.o ${PETSC_SNES_LIB}
	${RM} mp1.o
#--------------------------------------------------------------------------
runmp1:
	-@${MPIEXEC} -np 1 ./mp1 -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short > mp1_1.tmp 2>&1;   \
	   if (${DIFF} output/mp1_1.out mp1_1.tmp) then true; \
	   else echo "Possible problem with mp1_1, diffs above"; fi; \
	   ${RM} -f mp1_1.tmp

TESTEXAMPLES_C		       = mp1.PETSc runmp1 mp1.rm 
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
