# $Id: makefile,v 1.1 1999/10/06 16:37:49 bsmith Exp bsmith $ 

ALL: ex15 ex11 knoll knmg

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =	 
LOCDIR          = src/contrib/keyes/


include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

knmgcv: knmgcv.o chkopts
	-${CLINKER} -o knmgcv knmgcv.o ${PETSC_LIB}
	${RM} knmgcv.o

knoll: knoll.o chkopts
	-${CLINKER} -o knoll knoll.o ${PETSC_LIB}
	${RM} knoll.o

knmg: knmg.o chkopts
	-${CLINKER} -o knmg knmg.o ${PETSC_LIB}
	${RM} knmg.o

run32:
	-@${MPIEXEC} -n 1 knmgcv -mx 32 -my 32 -snes_monitor -snes_fd  -ncycles 50 -snes_rtol 5.0e-3 -ksp_rtol 1.0e-5 -dt 2.e-3 -on_error_attach_debugger nox -ksp_monitor -contours

run64:
	-@${MPIEXEC} -n 1 knmgcv -mx 64 -my 64 -snes_monitor -ksp_monitor -snes_fd  -ncycles 110 -snes_rtol 5.0e-3 -ksp_rtol 1.0e-5 -dt 2.e-3 -on_error_attach_debugger nox -ksp_monitor

runmgcv:
	-@${MPIEXEC} -n 1 knmgcv -mx 16 -my 16 -snes_monitor -snes_fd  -ncycles 1 -dt 1.e9 -on_error_attach_debugger nox -ksp_monitor

runmgcvunsteady:
	-@${MPIEXEC} -n 1 knmgcv -mx 16 -my 16 -snes_monitor -snes_fd  -ncycles 20 -dt 1.e-3 -on_error_attach_debugger nox -ksp_monitor

runmg:
	-@${MPIEXEC} -n 1 knmg -mx 16 -my 16 -snes_monitor -snes_fd  -ncycles 1 -dt 1.e9 -on_error_attach_debugger nox -ksp_monitor

runk:
	-@${MPIEXEC} -n 1 knoll -mx 6 -my 6 -snes_monitor -snes_fd  -ksp_monitor -ncycles 10 -dt 1.e-4 -on_error_attach_debugger nos


ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o ${PETSC_SNES_LIB}
	${RM} ex11.o

ex15: ex15.o  chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_SNES_LIB}
	${RM} ex15.o

ex17: ex17.o  chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_SNES_LIB}
	${RM} ex17.o

runex11s:
	-@${MPIEXEC} -n 1 ex11 -Mx 20 -My 20 -snes_monitor_short -ksp_monitor \
	-coarse_ksp_type preonly -coarse_pc_type lu \
	-fine_ksp_type richardson \
	-fine_ksp_max_it 1 \
	-pc_mg_type multiplicative \
	-ratio 2 \
	-optionsleft

runex11p:
	-@${MPIEXEC} -n 4 ex11 -Mx 20 -My 20 -snes_monitor_short -ksp_monitor \
	-coarse_pc_type redundant -coarse_ksp_type preonly \
	-coarse_redundant_pc_type lu \
	-coarse_redundant_ksp_type preonly \
	-fine_ksp_type richardson \
	-fine_ksp_max_it 1 -fine_pc_type asm \
	-pc_mg_type multiplicative \
	-ratio 2 \
	-optionsleft

runex15linear:
	-@${MPIEXEC} -n 1 ex15 -Mx 20 -My 20 -snes_monitor_short -ksp_monitor \
	-beta 0. -bm1 0. -coef 0. \
	-coarse_ksp_type preonly -coarse_pc_type lu \
	-fine_ksp_type richardson \
	-fine_ksp_max_it 1 \
	-snes_max_it 10 \
	-pc_mg_type multiplicative \
	-ratio 2 \
	-optionsleft

runex15quasi:
	-@${MPIEXEC} -n 1 ex15 -Mx 20 -My 20 -snes_monitor_short -ksp_monitor \
	-beta 1.5 -bm1 0.5 -coef 0.75 \
	-coarse_ksp_type preonly -coarse_pc_type lu \
	-fine_ksp_type richardson \
	-fine_ksp_max_it 1 \
	-snes_max_it 10 \
	-pc_mg_type multiplicative \
	-ratio 2 \
	-optionsleft

runex15:
	-@${MPIEXEC} -n 1 ex15 -Mx 10 -My 10 -snes_monitor_short -ksp_monitor \
	-coarse_ksp_type preonly -coarse_pc_type lu \
	-fine_ksp_type gmres \
	-fine_ksp_max_it 1 \
	-snes_max_it 20 \
	-pc_mg_type multiplicative \
	-ratio 3 -snes_rtol 1.0e-5 -ksp_rtol 5.0e-3 \
	-snes_view -optionsleft

runex15s:
	-@${MPIEXEC} -n 1 ex15 -Mx 20 -My 20 -snes_monitor_short -ksp_monitor \
	-coarse_ksp_type preonly -coarse_pc_type lu \
	-fine_ksp_type richardson \
	-fine_ksp_max_it 1 \
	-pc_mg_type multiplicative \
	-snes_rtol 1.0e-5 -ksp_rtol 5.0e-3 \
	-ratio 2 \
	-optionsleft

runex15p:
	-@${MPIEXEC} -n 4 ex15 -Mx 20 -My 20 -snes_monitor_short -ksp_monitor \
	-coarse_pc_type redundant -coarse_ksp_type preonly  \
	-coarse_redundant_pc_type lu \
	-coarse_redundant_ksp_type preonly \
	-fine_ksp_type richardson \
	-fine_ksp_max_it 1 -fine_pc_type asm \
	-pc_mg_type multiplicative \
	-snes_rtol 1.0e-5 -ksp_rtol 5.0e-3 \
	-ratio 2 \
	-optionsleft

runex15a:
	-@${MPIEXEC} -n 4 ex15 -Mx 17 -My 17 \
	-coarse_pc_type redundant -coarse_ksp_type preonly  \
	-coarse_redundant_pc_type lu \
	-coarse_redundant_ksp_type preonly \
	-fine_ksp_type richardson \
	-fine_ksp_max_it 1 -fine_pc_type asm \
	-mg_levels_sub_pc_factor_levels 1 \
	-pc_mg_type multiplicative \
	-snes_rtol 1.0e-5 -ksp_rtol 5.0e-3 \
	-ratio 4 -log_summary \
	-optionsleft

