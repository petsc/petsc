
CFLAGS           =
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = tutorials/multiphysics/
MANSEC           = SNES
EXAMPLESC	 = mp.c p1.c p2.c ff1.c ff2.c
EXAMPLESF	 = 
EXAMPLESCH	 = mp.h
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
	-@${MPIEXEC} -n 1 ./mp -grashof 1.0 -snes_monitor_short > mp_1.tmp 2>&1;   \
	   if (${DIFF} output/mp_1.out mp_1.tmp) then true; \
	   else echo "Possible problem with mp_1, diffs above"; fi; \
	   ${RM} -f mp_1.tmp

runmp_2:
	-@${MPIEXEC} -n 1 ./mp -grashof 1.0 -dmmg_jacobian_mf_fd_operator -snes_monitor_short > mp_2.tmp 2>&1;   \
	   if (${DIFF} output/mp_2.out mp_2.tmp) then true; \
	   else echo "Possible problem with mp_2, diffs above"; fi; \
	   ${RM} -f mp_2.tmp

runmp_3:
	-@${MPIEXEC} -n 1 ./mp -grashof 1.0 -dmmg_jacobian_mf_fd_operator -dmcomposite_dense_jacobian > mp_3.tmp 2>&1;   \
	   if (${DIFF} output/mp_3.out mp_3.tmp) then true; \
	   else echo "Possible problem with mp_3, diffs above"; fi; \
	   ${RM} -f mp_3.tmp

# this should have the same SNES convergence as _2 since they both are using the correct Jacobian
runmp_4:
	-@${MPIEXEC} -n 1 ./mp -grashof 1.0 -couple -snes_monitor_short  > mp_4.tmp 2>&1;   \
	   if (${DIFF} output/mp_4.out mp_4.tmp) then true; \
	   else echo "Possible problem with mp_4, diffs above"; fi; \
	   ${RM} -f mp_4.tmp

runmp_5:
	-@${MPIEXEC} -n 1 ./mp -grashof 1.0 -couple -snes_monitor_short  -ksp_view -pc_type fieldsplit > mp_5.tmp 2>&1;   \
	   if (${DIFF} output/mp_5.out mp_5.tmp) then true; \
	   else echo "Possible problem with mp_5, diffs above"; fi; \
	   ${RM} -f mp_5.tmp

runmp_6:
	-@${MPIEXEC} -n 1 ./mp -grashof 1.0 -couple -snes_monitor_short  -ksp_view -pc_type fieldsplit -snes_mf_operator > mp_6.tmp 2>&1;   \
	   if (${DIFF} output/mp_6.out mp_6.tmp) then true; \
	   else echo "Possible problem with mp_6, diffs above"; fi; \
	   ${RM} -f mp_6.tmp

runmp_7:
	-@${MPIEXEC} -n 3 ./mp -grashof 1.0 -couple -ksp_monitor_short  -pc_type fieldsplit > mp_7.tmp 2>&1;   \
	   if (${DIFF} output/mp_7.out mp_7.tmp) then true; \
	   else echo "Possible problem with mp_7, diffs above"; fi; \
	   ${RM} -f mp_7.tmp

runmp_8:
	-@${MPIEXEC} -n 3 ./mp -grashof 1.0 -couple -ksp_monitor_short  -pc_type fieldsplit -snes_mf_operator > mp_8.tmp 2>&1;   \
	   if (${DIFF} output/mp_8.out mp_8.tmp) then true; \
	   else echo "Possible problem with mp_8, diffs above"; fi; \
	   ${RM} -f mp_8.tmp

runmp_9:
	-@${MPIEXEC} -n 1 ./mp -couple -snes_monitor_short -pc_type fieldsplit -ksp_monitor_short -pc_fieldsplit_type schur -fieldsplit_1_ksp_monitor_short -fieldsplit_1_ksp_type fgmres -fieldsplit_0_ksp_type gmres -fieldsplit_0_ksp_monitor_short -pc_fieldsplit_schur_precondition diag -snes_view > mp_9.tmp 2>&1;   \
	   if (${DIFF} output/mp_9.out mp_9.tmp) then true; \
	   else echo "Possible problem with mp_9, diffs above"; fi; \
	   ${RM} -f mp_9.tmp

runmp_10:
	-@${MPIEXEC} -n 1 ./mp -couple -snes_monitor_short -pc_type fieldsplit -ksp_monitor_short -pc_fieldsplit_type schur -fieldsplit_ksp_monitor_short -fieldsplit_1_ksp_type fgmres -fieldsplit_0_ksp_type gmres -fieldsplit_0_ksp_monitor_short -pc_fieldsplit_schur_precondition self -snes_view > mp_10.tmp 2>&1;   \
	   if (${DIFF} output/mp_10.out mp_10.tmp) then true; \
	   else echo "Possible problem with mp_10, diffs above"; fi; \
	   ${RM} -f mp_10.tmp

TESTEXAMPLES_C		       = p1.PETSc runp1 p1.rm p2.PETSc runp2 p2.rm mp.PETSc runmp runmp_2 runmp_3 runmp_4 runmp_5 runmp_6 runmp_7 runmp_8 runmp_9 runmp_10 mp.rm
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
